import pytorch_lightning as pl
import wandb
from config_utils import get_config
from dataset.coco_utils import get_coco_api_from_dataset
from dataset.dataloader import get_dataloader
from mask_rcnn import MaskRCNN
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main() -> None:
    params = get_config("config.ini")
    pl.seed_everything(params.general.seed)

    train_loader, val_loader = get_dataloader(
        root=[
            params.data.train_data_root,
            params.data.val_data_root,
        ],
        annFile=[
            params.data.train_ann_file,
            params.data.val_ann_file,
        ],
        shuffle=params.loader.shuffle,
        resize_size=params.loader.resize_size,
        batch_size=params.loader.batch_size,
        num_workers=params.loader.num_workers,
    )

    coco = get_coco_api_from_dataset(train_loader.dataset)
    model = MaskRCNN(params=params, num_classes=len(coco.cats) + 1)  # 0 is background class

    wandb_logger = (
        WandbLogger(
            entity=params.logging.entity,
            project=params.logging.project,
            log_model=params.logging.log_model,
        )
        if params.logging.wandb
        else None
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val/map",
        mode="max",
        filename="{epoch}-{val/map:.2f}",
    )

    trainer = pl.Trainer(
        accelerator=params.general.device,
        max_epochs=20,
        logger=wandb_logger,  # type: ignore
        callbacks=[checkpoint_callback],
        log_every_n_steps=params.logging.log_every_n_steps,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if params.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
