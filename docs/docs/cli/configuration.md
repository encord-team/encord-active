---
sidebar_position: 5
---

# Configuration

Encord Active keeps some configurable properties to prevent repetitive input prompts.

The config file is stored at:

- Linux: `~/.config/encord-active/config.toml`
- MacOS: `~/Library/Application Support/encord-active/config.toml`
- Windows: `%APPDATA%/encord-active/config.toml`

And these are the configurable properties.

```toml
ssh_key_path = "/absolute/path/to/ssh-key" # A key to use when accessing Encord projects
projects_dir = "/absolute/path/to/projects/dir" # A directory where all projects should be stored
```

All properties are empty by default and are saved after the first time the user is prompted to provide them.

The CLI also provides the `config` command which allows `list`/`get`/`set`/`unset` configurations.

### Usage examples

```shell
encord-active config list

# output (only if already set)
ssh_key_path = "/Users/foo/.ssh/encord"
projects_dir = "/Users/foo/projects"
```

```shell
encord-active config get ssh_key_path

# output
ssh_key_path = "/Users/foo/.ssh/encord"
```

```shell
encord-active config set ssh_key_path ~/.ssh/encord

# output
Property `ssh_key_path` has been set.
```

```shell
encord-active config unset ssh_key_path

# output
Property `ssh_key_path` was unset.

# note: next the property is needed a prompt will appear
```
