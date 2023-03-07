from pathlib import Path
from typing import Literal, NamedTuple, Optional, Union

from git.objects import Commit
from git.repo import Repo

GITIGNORE = """data/**
embeddings/**
**/*.zip
"""


class Version(NamedTuple):
    name: str
    id: str


class GitVersioner:
    def __init__(self, path: Path) -> None:
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
    def _default_head(self):
        return self.repo.heads.__getattr__(self._default_branch)

    @property
    def current_version(self):
        return _commit_to_version(self.repo.head.commit)

    def is_latest(self, version: Optional[Version] = None) -> bool:
        return (version or self.current_version).id == self._default_head.commit.hexsha

    @property
    def versions(self):
        return [_commit_to_version(commit) for commit in self.repo.iter_commits(self._default_head)]

    @property
    def has_changes(self):
        return self.repo.is_dirty() or bool(self.repo.untracked_files)

    def create_version(self, name: str):
        if not self.is_latest():
            raise Exception("Creating versions is only allowed fron the latest version.")
        self.repo.git.add("-A")
        new_version = _commit_to_version(self.repo.index.commit(name))
        self.jump_to(new_version)
        return new_version

    def jump_to(self, version: Union[Version, Literal["latest"]]):
        if version == "latest" or version.id == self._default_head.commit.hexsha and not self.is_latest():
            self.repo.head.reference = self._default_head  # type: ignore
            self.discard_changes()
        elif self.repo.head.commit.hexsha != version.id:
            self.repo.head.reference = self.repo.rev_parse(version.id)  # type: ignore

    def discard_changes(self):
        self.repo.head.reset(index=True, working_tree=True)

    def stash(self):
        self.repo.git.stash("save", "", "--include-untracked")

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
        global_default = str(reader.get_value("init", "defaultBranch")).strip('"')

        return global_default if global_default in possible_heads else possible_heads.pop()
