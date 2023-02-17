---
sidebar_position: 4
---

# Custom Data Import

> 🔥 HARD 🔥 If you have a custom data and label structure.

In the situation where you have a custom data and label structure, you have two options:

1. [Migrate your data to Encord](/sdk/migrating-data) before [importing](./import-encord-project) it with  
   `encord-active import project`
2. Convert your data and labels to the [COCO data format][coco-format] before [importing](./import-coco-project) it with  
   `encord-active import project --coco -i ./images -a ./annotations.json`.

:::info

We are working hard on a smoother data onboarding process so you won't need an Encord account and won't have to ship your data anywhere.
If you want to know more, please contact us via the [Slack community][slack-invite] or by sending us an email on [active@encord.com](mailto:active@encord.com).

:::

[slack-invite]: https://join.slack.com/t/encordactive/shared_invite/zt-1hc2vqur9-Fzj1EEAHoqu91sZ0CX0A7Q
[coco-format]: https://cocodataset.org/#format-data
