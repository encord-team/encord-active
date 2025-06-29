---
title: "Contributing"
slug: "active-contributing"
hidden: false
metadata: 
  title: "Contributing"
  description: "Join our community: Contribute to Encord Active's growth. Improve docs, give feedback, share – stars on GitHub appreciated!"
  image: 
    0: "https://files.readme.io/c1f7df0-image_16.png"
createdAt: "2023-07-19T08:51:45.516Z"
updatedAt: "2023-08-09T11:49:15.064Z"
category: "65a71bbfea7a3f005192d1a7"
---
We follow a [code of conduct](https://github.com/encord-team/encord-active/blob/main/CODE_OF_CONDUCT.md) when participating in the community. Please read it before you make any contributions.

- If you plan to work on an issue, mention so in the issue page before you start working on it.
- If you plan to work on a new feature, create an issue and discuss it with other community members/maintainers.
- Ask for help in our [Slack community][slack-join].

## Ways to contribute

- **Stars on GitHub**: If you are an Encord Active user and enjoy using our platform, don't forget to star it on [GitHub](https://github.com/encord-team/encord-active)! 🌟
- **Improve documentation**: Good documentation is imperative to the success of any project. You can make our documents the best they need to be by improving their quality or adding new ones.
- **Give feedback**: We are always looking for ways to make Encord Active better, please share how you use Encord Active, what features are missing and what is done good via [Slack][slack-join].
- **Share refine**: Help us reach people. Share [Encord Active repository](https://github.com/encord-team/encord-active) with everyone who can be interested.
- **Contribute to codebase**: Your help is needed to make this project the best it can be! You could develop new features or fix [existing issues](https://github.com/encord-team/encord-active/issues) - all contributions are welcome!

## Environment setup

Make sure you have `python3.9` installed on your system.

To install the correct version of python you can use [pyenv](https://github.com/pyenv/pyenv), [brew (mac only)](https://formulae.brew.sh/formula/python@3.9) or simply [download](https://www.python.org/downloads) it.

You'll also need to have [poetry installed](https://python-poetry.org/docs/#installation).

After forking and cloning the repository, run:

```shell
poetry install

# If you intend to work on coco related things, run this instead:
poetry install --extras "coco"
```

> ℹ️ Note
> You might need to install `xcode-select` if you are on Mac or `C++ build tools` if you are on Windows.

After the installation is done, you can activate the created virtual environment with:

```shell
poetry shell
```

Now you should be able to run your locally installed `encord-active`.

> ℹ️ Note
> Make sure you are always running `encord-active` from the activated virtual environment to not conflict with a globally installed version.


### Running the frontend

> ℹ️ Note
> Running the frontend locally is only required if you intend to work on our React frontend components.

Our frontend is build with [React](https://reactjs.org/). To start it in development mode, run:

```shell
cd "frontend_components/encord_active_components/frontend" && npm i && npm start
```

In order to point `encord-active` to your locally running frontend, you'll need to change the `FRONTEND` environment variable in the `.env` file. Make sure you point it to the correct port, by default it should be running on `http://localhost:5173/`

## Commit convention

Commit messages are essential to make changes clear and concise. We use [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) to keep our commit messages consistent and easy to understand.

```
<type>(optional scope): <description>
```

Examples:

- `feat: allow provided config object to extend other configs`
- `fix: array parsing issue when multiple spaces were contained in string`
- `docs: correct spelling of CHANGELOG`

## Contribution guide

You need to follow these steps below to make contribution to the main repository via pull request. You can learn about the details of pull request [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

### 1. Fork the official repository

If you are using Git, you can visit the [Encord Active repository](https://github.com/encord-team/encord-active) and find the <kbd>Fork</kbd> button at the right top corner of the web page, along with other buttons such as <kbd>Watch</kbd> and <kbd>Star</kbd> (highly appreciated if you click this one as well 🌟). Simply click the <kbd>Fork</kbd> button to create a copy of the repository under your own account.

Now, you can clone your own forked repository into your local environment.

```shell
git clone https://github.com/<YOUR-USERNAME>/encord-active.git
```

Otherwise, if you have the GitHub CLI installed, the following command will create a fork. If you haven't, consider [installing it](https://github.com/cli/cli#installation).

```shell
gh repo fork encord-team/encord-active
```

### 2. Configure Git

You need to set the official repository as your upstream so that you can synchronize with the latest updates in the official repository. You can learn about syncing forks [here](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork).

With Git, it's as simple as running the following commands:

```shell
cd encord-active
git remote add upstream https://github.com/encord-team/encord-active.git
```

> If you use the GitHub CLI, this step is done automatically 🪄

You can use the following command to verify that the remote is set. You should see both `origin` and `upstream` in the output.

```shell
git remote -v
> origin    https://github.com/<YOUR-USERNAME>/encord-active.git (fetch)
> origin    https://github.com/<YOUR-USERNAME>/encord-active.git (push)
> upstream  https://github.com/encord-team/encord-active.git (fetch)
> upstream  https://github.com/encord-team/encord-active.git (push)
```

### 3. Synchronize

Before you make changes to the codebase, it is always good to fetch the latest updates in the official repository. In order to do so, you can use the commands below.

#### Git

```shell
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

Otherwise, you can click the `fetch upstream` button on the GitHub webpage of the main branch of your forked repository. Then, use these commands to sync.

```shell
git checkout main
git fetch main
```

#### GitHub CLI

To sync your remote fork:

```shell
gh repo sync <YOUR-USERNAME>/encord-active
```

And then to sync your local clone:

```shell
gh repo sync
```


### 4. Pull request issue

In order to not waste your time implementing a change that has already been declined, or is generally not needed, start by opening an [issue](https://github.com/encord-team/encord-active/issues) describing the problem you would like to solve. Make sure you use appropriate title and description and be as descriptive as possible.

Generally, your code change should be only targeting one problem in order to make the review process as simple as possible.

### 5. Make changes

You should not make changes to the `main` branch of your forked repository as this might make upstream synchronization difficult. You can create a new branch with the appropriate name. Generally, branch names should start with a conventional commit type, e.g. `fix/` / `docs/` / `feat/` followed by the scope.

```shell
git checkout -b <NEW-BRANCH-NAME>
```

It is finally time to implement your change!

You can commit and push the changes to your local repository. The changes should be kept logical, modular and atomic.

```shell
git add -A
git commit -m "<COMMIT-TYPE>: <COMMIT-MESSAGE>"
git push -u origin <NEW-BRANCH-NAME>
```

> 👍 Tip
> If you are making changes to the frontend, you can run `encord-active config set dev true` to enable file watchers. This will make the UI detect code changes and suggest to (auto) refresh the page.


### 7. Open a pull request

You can now create a pull request on the GitHub webpage of your repository. The source branch is `<NEW-BRANCH-NAME>` of your repository and the target branch should be `main` of `encord-team/encord-active`. After creating this pull request, you should be able to see it [here](https://github.com/encord-team/encord-active/pulls).

If you are using the GitHub CLI you can run:

```shell
gh pr create --web
```

Fill out the title and body appropriately trying to be as clear as possible. And again, make sure to follow the conventional commit guidelines for your title.

Do write a clear description of your pull request and [link the pull request to your target issue](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue). This will automatically close the issue when the pull request is approved.

In case of code conflict, you should rebase your branch and resolve the conflicts manually.


[slack-join]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
