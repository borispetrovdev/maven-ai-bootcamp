from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


class ItemPayload(BaseModel):
    preprocessed_description: str = Field(description="Cleaned item description text")
    image: HttpUrl = Field(description="Image URL of the item")
    rating_number: int = Field(ge=0, description="Number of ratings")
    price: float | None = Field(default=None, ge=0, description="Item price")
    average_rating: float = Field(ge=0, le=5, description="Average rating (0-5)")
    parent_asin: str = Field(description="Amazon parent ASIN identifier")


class RAGRequest(BaseModel):
    query: str


class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of the item used to answer the question")
    image_url: str = Field(description="Image URL of the item corresponding to the id")
    price: Optional[float] = Field(
        default=None, description="Price of the item corresponding to the id"
    )
    description: str = Field(
        description="Description of the item corresponding to the id"
    )


class RAGResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext] = Field(
        description="List of items used to answer the question"
    )
