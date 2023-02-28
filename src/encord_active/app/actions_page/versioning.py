import streamlit as st

from encord_active.app.common.state import get_state, refresh
from encord_active.app.common.state_hooks import use_memo, use_state
from encord_active.lib.versioning.git import GitVersioner

CURRENT_VERSION_KEY = "current_version"


def version_selector(versioner: GitVersioner):
    initial_version_index = use_memo(lambda: versioner.versions.index(versioner.current_version))
    get_version, set_version = use_state(versioner.versions[0], CURRENT_VERSION_KEY)
    version = st.selectbox(
        "Choose version", versioner.versions, format_func=lambda version: version.name, index=initial_version_index
    )

    if not version or version.id == get_version().id:
        return version

    if versioner.is_latest(get_version()):
        versioner.stash()

    set_version(version)
    versioner.jump_to(version)
    versioner.discard_changes()

    if versioner.is_latest(version):
        versioner.unstash()

    refresh(True)


def version_form():
    _, container, _ = st.columns(3)
    versioner = GitVersioner(get_state().project_paths.project_dir)
    _, set_version = use_state(versioner.versions[0], CURRENT_VERSION_KEY)

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
            refresh(True)
        if create.form_submit_button("Create", use_container_width=True, type="primary", **opts):
            if not version_name:
                st.error("Version name is required.")
            else:
                version = versioner.create_version(version_name)
                set_version(version)
                refresh()
