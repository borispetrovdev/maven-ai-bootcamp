from typing import Literal, TypedDict

import numpy as np
import psycopg2
from api.agents.retrieval_generation import (
    process_context,
    process_retrieved_reviews,
    rerank_data,
    retrieve_items_data,
    retrieve_prefiltered_reviews_data,
)
from api.api.models import ItemPayload
from langchain_core.tools import tool
from psycopg2.extras import RealDictCursor
from pydantic import BaseModel, Field, TypeAdapter
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)


@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """Search available products and return the top k matching inventory items.

    Expand the customer's question into 1-5 concise search statements and issue them
    in parallel in a single turn. Each statement covers one distinct product or
    attribute; no two may express the same intent. Use natural product-description
    language. If no brand or model is specified, search broadly rather than refusing.

        "Earphones for me and a waterproof speaker"
            -> "Personal earphones" | "Waterproof speaker"
        "A warm winter jacket for hiking"
            -> "Insulated winter jacket" | "Hiking outerwear for cold weather"

    Before calling, check what earlier calls in this conversation already returned.
    Search only for what is missing; results already retrieved remain valid and must
    not be fetched again.

    Args:
        query: A single search statement describing one product or attribute.
        top_k: Number of items to retrieve. Works best with 5 or more.

    Returns:
        A string of the top k available products, each prefixed with its ID and
        average rating.
    """

    qdrant_client = QdrantClient(url="http://localhost:6333")
    retrieved_context = retrieve_items_data(query, qdrant_client, k=20, hybrid=True)

    retrieved_context = rerank_data(query, retrieved_context, top_k=top_k)
    formatted_context = process_context(retrieved_context)
    return formatted_context


@tool
def get_formatted_reviews_context(
    query: str, parent_asins: list[str], top_k: int = 5
) -> str:
    """Get the top k reviews matching a query for a list of prefiltered items.

    Args:
        query: The query to get the top k reviews for
        item_list: The list of item IDs to prefilter for before running the query
        top_k: The number of reviews to retrieve, this should be at least 20 if multipple items are prefiltered

    Returns:
        A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for a given query.
    """

    qdrant_client = QdrantClient(url="http://localhost:6333")

    retrieved_context = retrieve_prefiltered_reviews_data(
        query, parent_asins, qdrant_client, k=top_k
    )

    formatted_context = process_retrieved_reviews(retrieved_context)
    return formatted_context


### Shopping Cart Tools


class ProductInDb(TypedDict):
    product_id: str
    quantity: int


items: list[ProductInDb] = [
    ProductInDb(product_id="B0BWJMKC31", quantity=2),
    ProductInDb(product_id="B0BT9PWL81", quantity=4),
]


@tool
def add_to_shopping_cart(items: list[ProductInDb], user_id: str, cart_id: str) -> str:
    """Add a list of provided items to the shopping cart.

    Args:
        items: A list of items to add to the shopping cart. Each item is a dictionary with the following keys: product_id, quantity.
        user_id: The id of the user to add the items to the shopping cart.
        cart_id: The id of the shopping cart to add the items to.

    Returns:
        A list of the items added to the shopping cart.
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        for item in items:
            product_id = item["product_id"]
            quantity = item["quantity"]

            qdrant_client = QdrantClient(url="http://localhost:6333")

            dummy_vector = np.zeros(1536).tolist()
            payload = ItemPayload.model_validate(
                qdrant_client.query_points(
                    collection_name="Amazon-items-collection-01-hybrid-search",
                    prefetch=[
                        Prefetch(
                            query=dummy_vector,
                            filter=Filter(
                                must=[
                                    FieldCondition(
                                        key="parent_asin",
                                        match=MatchValue(value=product_id),
                                    )
                                ]
                            ),
                            using="text-embedding-3-small",
                            limit=20,
                        )
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=1,
                )
                .points[0]
                .payload
            )

            product_image_url = str(payload.image)
            price = payload.price
            currency = "USD"

            # Check if item already exists
            check_query = """
                SELECT id, quantity, price 
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
            cursor.execute(check_query, (user_id, cart_id, product_id))
            existing_item = cursor.fetchone()

            if existing_item:
                # Update existing item
                new_quantity = existing_item["quantity"] + quantity

                update_query = """
                    UPDATE shopping_carts.shopping_cart_items 
                    SET 
                        quantity = %s,
                        price = %s,
                        currency = %s,
                        product_image_url = COALESCE(%s, product_image_url)
                    WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
                    RETURNING id, quantity, price
                """

                cursor.execute(
                    update_query,
                    (
                        new_quantity,
                        price,
                        currency,
                        product_image_url,
                        user_id,
                        cart_id,
                        product_id,
                    ),
                )

            else:
                # Insert new item
                insert_query = """
                    INSERT INTO shopping_carts.shopping_cart_items (
                        user_id, shopping_cart_id, product_id,
                        price, quantity, currency, product_image_url
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, quantity, price
                """

                cursor.execute(
                    insert_query,
                    (
                        user_id,
                        cart_id,
                        product_id,
                        price,
                        quantity,
                        currency,
                        product_image_url,
                    ),
                )

    return f"Added {items} to the shopping cart."


@tool
def remove_from_cart(product_id: str, user_id: str, cart_id: str) -> str:
    """
    Remove an item completely from the shopping cart.

    Args:
        user_id: User identifier
        product_id: Product identifier to remove
        cart_id: Cart identifier

    Returns:
        Information about the removal of the item from the shopping cart.
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
                DELETE FROM shopping_carts.shopping_cart_items
                WHERE user_id = %s AND shopping_cart_id = %s AND product_id = %s
            """
        cursor.execute(query, (user_id, cart_id, product_id))

        return (
            f"Removed {product_id} from the shopping cart."
            if cursor.rowcount > 0
            else f"Item {product_id} not found in the shopping cart."
        )


_product_type_adapter = TypeAdapter(ProductInDb)


@tool
def get_shopping_cart(user_id: str, cart_id: str) -> list[ProductInDb]:
    """
    Retrieve all items in a user's shopping cart.

    Args:
        user_id: User identifier
        cart_id: Cart identifier

    Returns:
        List of dictionaries containing cart items
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password",
    )
    conn.autocommit = True

    with conn.cursor(cursor_factory=RealDictCursor) as cursor:
        query = """
                SELECT 
                    product_id, price, quantity,
                    currency, product_image_url,
                    (price * quantity) as total_price
                FROM shopping_carts.shopping_cart_items 
                WHERE user_id = %s AND shopping_cart_id = %s
                ORDER BY added_at DESC
            """
        cursor.execute(query, (user_id, cart_id))

        return [
            _product_type_adapter.validate_python(dict(row))
            for row in cursor.fetchall()
        ]


class CartItem(BaseModel):
    """A single line in the shopping cart the warehouse tools operate on.

    A model rather than a TypedDict: this is the tool's input, so the LLM fills
    it in. The field descriptions become part of the JSON schema the model reads,
    and the values are validated at construction instead of trusted.
    """

    product_id: str = Field(description="The parent ASIN of the product to check.")
    quantity: int = Field(gt=0, description="How many units the customer wants.")


class WarehouseSummary(TypedDict):
    """Identifying fields of a warehouse, without any availability numbers."""

    warehouse_id: str
    warehouse_name: str
    warehouse_location: str


class ItemAvailability(TypedDict):
    """How much of one requested item a single warehouse can supply."""

    product_id: str
    requested: int
    available: int
    can_fulfill_completely: bool
    can_fulfill_partially: bool


class WarehouseAvailability(WarehouseSummary):
    """Per-warehouse breakdown: every requested item plus the warehouse verdict."""

    items: list[ItemAvailability]
    can_fulfill_all: bool
    has_partial: bool


class UnavailableItem(TypedDict):
    """An item whose requested quantity exceeds total stock across all warehouses."""

    product_id: str
    requested: int
    total_available_across_warehouses: int
    shortage: int


class AvailabilityCheck(TypedDict):
    """Result of checking a cart against every warehouse."""

    can_fulfill_completely: bool
    warehouses_full_fulfillment: list[WarehouseSummary]
    warehouses_partial_fulfillment: list[WarehouseSummary]
    unavailable_items: list[UnavailableItem]
    details: list[WarehouseAvailability]


class ReservationRequest(BaseModel):
    """One warehouse-specific reservation the LLM asks the tool to make.

    Unlike CartItem this names a warehouse, so the caller has already decided
    where each item comes from - typically from a check_warehouse_availability
    result.
    """

    warehouse_id: str = Field(description="The warehouse to reserve from.")
    product_id: str = Field(description="The parent ASIN of the product to reserve.")
    quantity: int = Field(gt=0, description="How many units to reserve.")


class ReservedItem(TypedDict):
    """An item successfully reserved at one warehouse."""

    product_id: str
    quantity: int
    warehouse_id: str
    warehouse_name: str
    warehouse_location: str


class FailedReservation(TypedDict):
    """An item that could not be reserved, and why."""

    product_id: str
    warehouse_id: str
    requested: int
    available: int
    reason: Literal["insufficient_stock", "not_in_warehouse"]


class ReservationSucceeded(TypedDict):
    """Every requested item was reserved and the transaction committed."""

    status: Literal["reserved"]
    reserved_items: list[ReservedItem]


class ReservationFailed(TypedDict):
    """At least one item could not be reserved, so the whole batch rolled back.

    There is deliberately no reserved_items key: the reservation is all-or-nothing,
    so on this branch nothing is held, however far the transaction got before the
    failing item.
    """

    status: Literal["rejected"]
    failed_items: list[FailedReservation]


# Discriminated on the literal "status" - narrow with an == comparison, since a
# truthiness check on a TypedDict subscript does not narrow the union.
ReservationResult = ReservationSucceeded | ReservationFailed


@tool
def check_warehouse_availability(items: list[CartItem]) -> AvailabilityCheck:
    """Check availability of items across warehouses, including partial fulfillment options.

    Args:
        items: A list of items to check. Each item is a CartItem with a product_id
            and a quantity.

    Returns:
        A dictionary containing:
        - can_fulfill_completely: bool indicating if all items can be fulfilled from at least one warehouse
        - warehouses_full_fulfillment: list of warehouses that can fulfill the entire order
        - warehouses_partial_fulfillment: list of warehouses with partial availability
        - unavailable_items: list of items that cannot be fulfilled from any warehouse
        - details: detailed breakdown per warehouse with availability for each item
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password",
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            result: AvailabilityCheck = {
                "can_fulfill_completely": False,
                "warehouses_full_fulfillment": [],
                "warehouses_partial_fulfillment": [],
                "unavailable_items": [],
                "details": [],
            }

            # Check each warehouse for availability
            warehouse_query = """
                SELECT DISTINCT warehouse_id, warehouse_name, warehouse_location
                FROM warehouses.inventory
            """
            cursor.execute(warehouse_query)
            warehouses = cursor.fetchall()

            for warehouse in warehouses:
                warehouse_can_fulfill_all = True
                has_any_availability = False
                warehouse_details: WarehouseAvailability = {
                    "warehouse_id": warehouse["warehouse_id"],
                    "warehouse_name": warehouse["warehouse_name"],
                    "warehouse_location": warehouse["warehouse_location"],
                    "items": [],
                    "can_fulfill_all": False,
                    "has_partial": False,
                }

                for item in items:
                    product_id = item.product_id
                    requested_quantity = item.quantity

                    # Check availability in this warehouse
                    availability_query = """
                        SELECT product_id, total_quantity, reserved_quantity, available_quantity
                        FROM warehouses.inventory
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(
                        availability_query, (warehouse["warehouse_id"], product_id)
                    )
                    inventory = cursor.fetchone()

                    # int() at the boundary: psycopg2 hands back Any, so this is the
                    # only place the declared int types are actually enforced.
                    available_qty = (
                        int(inventory["available_quantity"]) if inventory else 0
                    )

                    item_detail: ItemAvailability = {
                        "product_id": product_id,
                        "requested": requested_quantity,
                        "available": available_qty,
                        "can_fulfill_completely": available_qty >= requested_quantity,
                        "can_fulfill_partially": available_qty > 0
                        and available_qty < requested_quantity,
                    }

                    warehouse_details["items"].append(item_detail)

                    # Track if warehouse can fulfill this item completely
                    if available_qty < requested_quantity:
                        warehouse_can_fulfill_all = False

                    # Track if warehouse has any availability for any item
                    if available_qty > 0:
                        has_any_availability = True

                # Categorize warehouse
                if warehouse_can_fulfill_all:
                    warehouse_details["can_fulfill_all"] = True
                    result["warehouses_full_fulfillment"].append(
                        {
                            "warehouse_id": warehouse["warehouse_id"],
                            "warehouse_name": warehouse["warehouse_name"],
                            "warehouse_location": warehouse["warehouse_location"],
                        }
                    )
                elif has_any_availability:
                    warehouse_details["has_partial"] = True
                    result["warehouses_partial_fulfillment"].append(
                        {
                            "warehouse_id": warehouse["warehouse_id"],
                            "warehouse_name": warehouse["warehouse_name"],
                            "warehouse_location": warehouse["warehouse_location"],
                        }
                    )

                result["details"].append(warehouse_details)

            # Check if any items cannot be fulfilled from any warehouse
            for item in items:
                product_id = item.product_id
                requested_quantity = item.quantity

                # Get total available quantity across all warehouses
                total_available_query = """
                    SELECT product_id, SUM(available_quantity) as total_available
                    FROM warehouses.inventory
                    WHERE product_id = %s
                    GROUP BY product_id
                """
                cursor.execute(total_available_query, (product_id,))
                total_available = cursor.fetchone()

                # SUM() comes back as Decimal for a numeric column, so coerce here too.
                total_available_qty = (
                    int(total_available["total_available"]) if total_available else 0
                )

                if total_available_qty < requested_quantity:
                    result["unavailable_items"].append(
                        {
                            "product_id": product_id,
                            "requested": requested_quantity,
                            "total_available_across_warehouses": total_available_qty,
                            "shortage": requested_quantity - total_available_qty,
                        }
                    )

            result["can_fulfill_completely"] = (
                len(result["warehouses_full_fulfillment"]) > 0
                and len(result["unavailable_items"]) == 0
            )

            return result

    finally:
        conn.close()


@tool
def reserve_warehouse_items(
    reservations: list[ReservationRequest],
) -> ReservationResult:
    """Reserve items from multiple warehouses in a single transaction.

    Args:
        reservations: A list of reservations. Each reservation is a ReservationRequest
            naming the warehouse to reserve from, the product, and the quantity.

    Returns:
        Either a ReservationSucceeded with status "reserved" and the committed
        reserved_items, or a ReservationFailed with status "rejected" and the
        failed_items that caused the rollback. The reservation is all-or-nothing,
        so the rejected branch carries no reserved items at all.
    """

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="tools_database",
        user="tools_user",
        password="tools_user_password",
    )
    conn.autocommit = False  # Use transaction

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            reserved_items: list[ReservedItem] = []
            failed_items: list[FailedReservation] = []

            for reservation in reservations:
                warehouse_id = reservation.warehouse_id
                product_id = reservation.product_id
                quantity = reservation.quantity

                # Check and lock the inventory row
                check_query = """
                    SELECT warehouse_id, product_id, warehouse_name, warehouse_location, 
                           total_quantity, reserved_quantity, available_quantity
                    FROM warehouses.inventory
                    WHERE warehouse_id = %s AND product_id = %s
                    FOR UPDATE
                """
                cursor.execute(check_query, (warehouse_id, product_id))
                inventory = cursor.fetchone()

                # int()/str() at the boundary: psycopg2 hands back Any, so this is the
                # only place the declared types are actually enforced.
                available_qty = int(inventory["available_quantity"]) if inventory else 0

                if inventory and available_qty >= quantity:
                    # Update inventory to reserve the items
                    update_query = """
                        UPDATE warehouses.inventory
                        SET reserved_quantity = reserved_quantity + %s
                        WHERE warehouse_id = %s AND product_id = %s
                    """
                    cursor.execute(update_query, (quantity, warehouse_id, product_id))

                    reserved_items.append(
                        {
                            "product_id": product_id,
                            "quantity": quantity,
                            "warehouse_id": warehouse_id,
                            "warehouse_name": str(inventory["warehouse_name"]),
                            "warehouse_location": str(inventory["warehouse_location"]),
                        }
                    )
                else:
                    failed_items.append(
                        {
                            "product_id": product_id,
                            "warehouse_id": warehouse_id,
                            "requested": quantity,
                            "available": available_qty,
                            "reason": "insufficient_stock"
                            if inventory
                            else "not_in_warehouse",
                        }
                    )

            # Only commit if all items were successfully reserved. Anything appended
            # to reserved_items before a failure is rolled back with the rest, so it
            # must not appear in the returned payload.
            if failed_items:
                conn.rollback()
                return {"status": "rejected", "failed_items": failed_items}

            conn.commit()
            return {"status": "reserved", "reserved_items": reserved_items}

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
