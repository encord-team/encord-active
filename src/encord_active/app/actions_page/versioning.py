from pathlib import Path

import streamlit as st

from encord_active.app.common.state import get_state, refresh
from encord_active.app.common.state_hooks import UseState
from encord_active.lib.versioning.git import GitVersioner, Version

CURRENT_VERSION_KEY = "current_version"


@st.cache_resource(show_spinner=False)
def cached_versioner(project_path: Path):
    versioner = GitVersioner(project_path)
    index = versioner.versions.index(versioner.current_version)
    return versioner, index


def is_latest(project_path: Path):
    versioner, _ = cached_versioner(project_path)
    return versioner.is_latest()


def version_selector(project_path: Path):
    versioner, initial_version_index = cached_versioner(project_path)
    version_state = UseState[Version](versioner.versions[initial_version_index], CURRENT_VERSION_KEY)

    version = st.selectbox(
        "Choose version",
        versioner.versions,
        # NOTE: streamlit auto key generation is not smart enough to know the
        # options have changed. removing the key would lead to state issues.
        key=f"version-{project_path}",
        format_func=lambda version: version.name,
        index=initial_version_index,
    )

    if get_state() and get_state().project_paths.project_dir != project_path:
        version_state.set(versioner.current_version)

    if not version or version.id == version_state.value.id:
        return version

    if versioner.is_latest(version_state.value):
        versioner.stash()

    version_state.set(version)
    versioner.jump_to(version)
    versioner.discard_changes()

    if versioner.is_latest(version):
        versioner.unstash()

    refresh(clear_global=True)


def version_form():
    _, container, _ = st.columns(3)
    versioner = GitVersioner(get_state().project_paths.project_dir)
    version_state = UseState(versioner.versions[0], CURRENT_VERSION_KEY)
    show_success_message = UseState(False)

    if show_success_message.value:
        container.success(f'Successfully created version with name "{versioner.versions[0].name}"')
        show_success_message.set(False)

    with container:
        version_selector(get_state().project_paths.project_dir)

    opts = {}
    if not versioner.is_latest():
        msg = "Versioning is disabled in read only mode, change to the latest version."
        container.error(msg)
        opts = {"disabled": True, "help": msg}
    elif not versioner.has_changes:
        msg = "No changes to be versioned"
        container.info(msg)
        opts = {"disabled": True, "help": msg}

    with container.form("version-creation", True):
        version_name = st.text_input("Create new version", placeholder="Enter new version", **opts).strip()
        _, discard, create, _ = st.columns([1, 2, 2, 1])

        if discard.form_submit_button("Discard", use_container_width=True, **opts):
            versioner.discard_changes()
            refresh(clear_global=True)
        if create.form_submit_button("Create", use_container_width=True, type="primary", **opts):
            if not version_name:
                st.error("Version name is required.")
            else:
                version = versioner.create_version(version_name)
                version_state.set(version)
                show_success_message.set(True)
                refresh()
