import yaml
from jinja2 import Template
from langsmith import Client

ls_client = Client()


def prompt_template_config(yaml_path: str, prompt_key: str) -> Template:
    """Load a prompt from a YAML file and return it compiled, ready to `.render()`."""
    with open(yaml_path, "r") as file:
        config = yaml.safe_load(file)

    template_content = config["prompts"][prompt_key]
    return Template(template_content)


def prompt_template_registry(prompt_name: str) -> Template:
    """Pull a prompt from the LangSmith registry and return it compiled."""
    template_content = ls_client.pull_prompt(prompt_name).messages[0].prompt.template
    return Template(template_content)
