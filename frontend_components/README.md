# Encord Active Components

Frontend components used by Encord Active.
Built with React, Typescript, Vite and Tailwind.

### Publishing a new version

After making changes to the frontend side of things we need to build it.

```shell
cd src/encord_active_components/frontend
npm run build
```

When the frontend build is done, it is time to build the package and publish it.

Navigate back to the root directory

```shell
cd ../../../   # or `cd -` (anything that will get you back to root)
```

Bump the version

```shell
poetry version [patch|minor|major]
```

Build the package

```shell
poetry build
```

Publish the built package

```shell
# if this is your first time publising, first configure poetry
poetry config pypi-token.pypi <PYPY_TOKEN>

poetry publish
```

After publishing, we need to update the dependency in the the main project.

From the root of Encord Active project run:

```shell
poetry add encord-active-components@latest
```

This will update both `pyproject.toml` and `poetry.lock` to the latest version available.
