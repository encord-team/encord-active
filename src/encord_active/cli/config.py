import rich
import toml
import typer

from encord_active.app.app_config import CONFIG_PROPERTIES, app_config

config_cli = typer.Typer(rich_markup_mode="markdown")


@config_cli.command()
def list():
    """
    List Encord Active configuration properties.
    """
    rich.print(toml.dumps(app_config.contents) or "[bold red]Nothing configured.")


def _check_property(property: str):
    if property not in CONFIG_PROPERTIES:
        rich.print(f"[bold red]`{property}` is not a valid property.")
        rich.print("Valid properties are:")
        rich.print(CONFIG_PROPERTIES)
        exit()


@config_cli.command()
def get(
    property: str = typer.Argument(..., help="Name of the property"),
):
    """
    Print the value of an Encord Active configuration property.
    """
    _check_property(property)
    value = app_config.contents.get(property)
    rich.print(f"{property} = {value}" or f"[bold red]Property `{property}` not configured.")


@config_cli.command()
def set(
    property: str = typer.Argument(..., help="Name of the property"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """
    Sets an Encord Active configuration property.
    """
    _check_property(property)
    app_config.contents[property] = value
    app_config.save()

    rich.print(f"[bold green]Property `{property}` has been set.")


@config_cli.command()
def unset(
    property: str = typer.Argument(..., help="Name of the property"),
):
    """
    Unsets the value of an Encord Active configuration property.
    """
    _check_property(property)
    del app_config.contents[property]
    app_config.save()

    rich.print(f"[bold green]Property `{property}` was unset.")
