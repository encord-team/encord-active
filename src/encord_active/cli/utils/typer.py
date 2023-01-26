"""
Overwrite the `typer.rich_utils.rich_format_help` command to change the order 
of the default options.
"""

from collections import defaultdict
from typing import DefaultDict, List, Union

import click
import typer.rich_utils as rutils
from rich.align import Align
from rich.padding import Padding
from typer.core import MarkupMode


def rich_format_help_ea(
    *,
    obj: Union[click.Command, click.Group],
    ctx: click.Context,
    markup_mode: MarkupMode,
) -> None:
    """Print nicely formatted help text using rich.

    Based on original code from rich-cli, by @willmcgugan.
    https://github.com/Textualize/rich-cli/blob/8a2767c7a340715fc6fbf4930ace717b9b2fc5e5/src/rich_cli/__main__.py#L162-L236

    Replacement for the click function format_help().
    Takes a command or group and builds the help text output.
    """
    console = rutils._get_rich_console()

    console.print(
        Padding(rutils.highlighter(obj.get_usage(ctx)), 1),
        style=rutils.STYLE_USAGE_COMMAND,
    )

    # Print command / group help if we have some
    if obj.help:
        # Print with some padding
        console.print(
            Padding(
                Align(
                    rutils._get_help_text(
                        obj=obj,
                        markup_mode=markup_mode,
                    ),
                    pad=False,
                ),
                (0, 1, 1, 1),
            )
        )

    panel_to_arguments: DefaultDict[str, List[click.Argument]] = defaultdict(list)
    panel_to_options: DefaultDict[str, List[click.Option]] = defaultdict(list)

    for param in obj.get_params(ctx):
        # Skip if option is hidden
        if getattr(param, "hidden", False):
            continue
        if isinstance(param, click.Argument):
            panel_name = getattr(param, rutils._RICH_HELP_PANEL_NAME, None) or rutils.ARGUMENTS_PANEL_TITLE
            panel_to_arguments[panel_name].append(param)
        elif isinstance(param, click.Option):
            panel_name = getattr(param, rutils._RICH_HELP_PANEL_NAME, None) or rutils.OPTIONS_PANEL_TITLE
            panel_to_options[panel_name].append(param)

    default_arguments = panel_to_arguments.get(rutils.ARGUMENTS_PANEL_TITLE, [])

    rutils._print_options_panel(
        name=rutils.ARGUMENTS_PANEL_TITLE,
        params=default_arguments,
        ctx=ctx,
        markup_mode=markup_mode,
        console=console,
    )

    for panel_name, arguments in panel_to_arguments.items():
        if panel_name == rutils.ARGUMENTS_PANEL_TITLE:
            # Already printed above
            continue
        rutils._print_options_panel(
            name=panel_name,
            params=arguments,
            ctx=ctx,
            markup_mode=markup_mode,
            console=console,
        )

    for panel_name, options in panel_to_options.items():
        if panel_name == rutils.OPTIONS_PANEL_TITLE:
            # Already printed above
            continue
        rutils._print_options_panel(
            name=panel_name,
            params=options,
            ctx=ctx,
            markup_mode=markup_mode,
            console=console,
        )

    # MOVED DOWN
    # default_options = panel_to_options.get(rutils.OPTIONS_PANEL_TITLE, [])
    # rutils._print_options_panel(
    #     name=rutils.OPTIONS_PANEL_TITLE,
    #     params=default_options,
    #     ctx=ctx,
    #     markup_mode=markup_mode,
    #     console=console,
    # )
    # MOVE END

    if isinstance(obj, click.MultiCommand):
        panel_to_commands: DefaultDict[str, List[click.Command]] = defaultdict(list)
        for command_name in obj.list_commands(ctx):
            command = obj.get_command(ctx, command_name)
            if command and not command.hidden:
                panel_name = getattr(command, rutils._RICH_HELP_PANEL_NAME, None) or rutils.COMMANDS_PANEL_TITLE
                panel_to_commands[panel_name].append(command)

        # Print each command group panel
        default_commands = panel_to_commands.get(rutils.COMMANDS_PANEL_TITLE, [])
        rutils._print_commands_panel(
            name=rutils.COMMANDS_PANEL_TITLE,
            commands=default_commands,
            markup_mode=markup_mode,
            console=console,
        )
        for panel_name, commands in panel_to_commands.items():
            if panel_name == rutils.COMMANDS_PANEL_TITLE:
                # Already printed above
                continue
            rutils._print_commands_panel(
                name=panel_name,
                commands=commands,
                markup_mode=markup_mode,
                console=console,
            )

    # MOVED START
    default_options = panel_to_options.get(rutils.OPTIONS_PANEL_TITLE, [])
    rutils._print_options_panel(
        name=rutils.OPTIONS_PANEL_TITLE,
        params=default_options,
        ctx=ctx,
        markup_mode=markup_mode,
        console=console,
    )
    # MOVE END

    # Epilogue if we have it
    if obj.epilog:
        # Remove single linebreaks, replace double with single
        lines = obj.epilog.split("\n\n")
        epilogue = "\n".join([x.replace("\n", " ").strip() for x in lines])
        epilogue_text = rutils._make_rich_rext(text=epilogue, markup_mode=markup_mode)
        console.print(Padding(Align(epilogue_text, pad=False), 1))


rutils.rich_format_help = rich_format_help_ea
