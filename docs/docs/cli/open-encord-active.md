---
sidebar_position: 6
---

import Link from '@docusaurus/Link';

# Open Encord Active

:::info

If you haven't [downloaded](download-sandbox-data) or [imported](import-encord-project) a project yet, you need to do that first.

:::

To run the Encord Active app, you need the data path that you specified when you
[downloaded](download-sandbox-data) or [imported](import-encord-project) your project.

Run the following command:

```shell
# from the projects root
encord-active visualise

# from anywhere
encord-active visualise --target /path/to/project/root
```

Now, you will be able to select either of the projects you have previously imported or downloaded.
Upon hitting <key>enter</key>, your browser should open a new window with Encord Active.

:::info

If it is the first time you run the app, you may be asked for an email.
If you don't feel like sharing it, just hit <kbd>enter</kbd> to ignore it.

:::

:::caution

If the script just seems to hang and nothing happens in your browser, try visiting <Link to={"http://localhost:8501"}>http://localhost:8501</Link>.

:::
