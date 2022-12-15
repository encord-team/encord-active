from enum import Enum
from pathlib import Path

import rich
import toml
import typer
from rich.panel import Panel


class ConfigProperties(str, Enum):
    SSH_KEY_PATH = "ssh_key_path"


CONFIG_PROPERTIES = [p.value for p in ConfigProperties]


class AppConfig:
    def __init__(self, app_name: str):
        self.app_name = app_name

        app_dir = Path(typer.get_app_dir(self.app_name))
        if not app_dir.is_dir():
            app_dir.mkdir(parents=True)

        self.config_file = app_dir / "config.toml"
        self.load()

    def save(self):
        with open(self.config_file, "w", encoding="utf-8") as f:
            toml.dump(self.contents, f)

    def load(self):
        if not self.config_file.is_file():
            self.config_file.touch()

        self.contents = toml.load(self.config_file)

    def get_or_query_ssh_key(self) -> Path:
        if ConfigProperties.SSH_KEY_PATH.value not in self.contents:
            panel = Panel(
                """
Encord Active needs to know the path to the [blue]private ssh key[/blue] which is associated with Encord.
Don't know this? Please see our documentation on the topic to get more help.

[blue]https://docs.encord.com/admins/settings/public-keys/#set-up-public-key-authentication[/blue]
                """,
                title="SSH Key Path",
                expand=False,
            )

            rich.print(panel)

            ssh_key_path_str = typer.prompt("Where is your private ssh key stored?")
            ssh_key_path: Path = Path(ssh_key_path_str).expanduser().absolute()

            if not ssh_key_path.exists():
                rich.print(f"[red]The provided path `{ssh_key_path}` does not seem to be correct.")
                typer.Abort()

            self.contents[ConfigProperties.SSH_KEY_PATH.value] = ssh_key_path.as_posix()
            self.save()
        else:
            ssh_key_path = Path(self.contents[ConfigProperties.SSH_KEY_PATH.value]).expanduser()

        return ssh_key_path
