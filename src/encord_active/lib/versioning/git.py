from __future__ import annotations

from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, NamedTuple, Optional, Union

from git.objects import Commit
from git.repo import Repo

GITIGNORE = """data/**
embeddings/**
**/*.zip
"""


class GitNotAvailableError(Exception):
    ...


class Version(NamedTuple):
    name: str
    id: str


def check_availability(fn):
    @wraps(fn)
    def _check(_self: GitVersioner, *args, **kwargs):
        if not _self.available:
            raise GitNotAvailableError()
        return fn(_self, *args, **kwargs)

    return _check


class GitVersioner:
    def __init__(self, path: Path) -> None:
        self._set_is_versioning_available()
        if not self.available:
            return

        if (path / ".git").exists():
            self.repo = Repo(path)
        else:
            self.repo = Repo.init(path)
            with self.repo.config_writer() as writer:
                writer.set_value('diff "sqlite3"', "textconv", '"f() { sqlite3 \\"$@\\" .dump; }; f"')
            (path / ".git/info/attributes").write_text("sqlite.db diff=sqlite3")
            (path / ".gitignore").write_text(GITIGNORE)
            self.repo.index.add(self.repo.untracked_files)
            self.repo.index.commit("init")

        self._default_branch = _get_default_branch(self.repo)

    @property
    @check_availability
    def _default_head(self):
        return self.repo.heads.__getattr__(self._default_branch)

    @property
    @check_availability
    def current_version(self):
        return _commit_to_version(self.repo.head.commit)

    @check_availability
    def is_latest(self, version: Optional[Version] = None) -> bool:
        return (version or self.current_version).id == self._default_head.commit.hexsha

    @property
    @check_availability
    def versions(self):
        return [_commit_to_version(commit) for commit in self.repo.iter_commits(self._default_head)]

    @property
    @check_availability
    def has_changes(self):
        return self.repo.is_dirty() or bool(self.repo.untracked_files)

    def _set_is_versioning_available(self):
        self._available = True
        try:
            with TemporaryDirectory() as d:
                Repo.init(Path(d))
        except OSError:
            self._available = False

    @property
    def available(self):
        return self._available

    @check_availability
    def create_version(self, name: str):
        if not self.is_latest():
            raise Exception("Creating versions is only allowed fron the latest version.")
        self.repo.git.add("-A")
        new_version = _commit_to_version(self.repo.index.commit(name))
        self.jump_to(new_version)
        return new_version

    @check_availability
    def jump_to(self, version: Union[Version, Literal["latest"]]):
        if version == "latest" or version.id == self._default_head.commit.hexsha:
            if not self.is_latest():
                self.repo.head.reference = self._default_head  # type: ignore
                self.discard_changes()
        elif self.repo.head.commit.hexsha != version.id:
            self.repo.head.reference = self.repo.rev_parse(version.id)  # type: ignore

    @check_availability
    def discard_changes(self):
        self.repo.head.reset(index=True, working_tree=True)

    @check_availability
    def stash(self):
        self.repo.git.stash("save", "", "--include-untracked")

    @check_availability
    def unstash(self):
        try:
            self.repo.git.stash("pop")
        except:
            pass


def _commit_to_version(commit: Commit) -> Version:
    return Version(name=str(commit.message), id=commit.hexsha)


def _get_default_branch(repo: Repo):
    with repo.config_reader() as reader:
        possible_heads = [head.name for head in repo.heads]
        global_default = str(reader.get_value("init", "defaultBranch", "main")).strip('"')

        return global_default if global_default in possible_heads else possible_heads.pop()
