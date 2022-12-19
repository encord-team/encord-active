import pytorch_lightning as pl
import wandb
from config_utils import get_config
from dataset.coco_utils import get_coco_api_from_dataset
from dataset.dataloader import get_dataloader
from mask_rcnn import MaskRCNN
from pytorch_lightning.loggers import WandbLogger


def main() -> None:
    params = get_config("config.ini")
    pl.seed_everything(params.general.seed)

    test_loader = get_dataloader(
        root=params.data.test_data_root,
        annFile=params.data.test_ann_file,
        resize_size=params.loader.resize_size,
        batch_size=params.loader.batch_size,
        num_workers=params.loader.num_workers,
    )

    coco = get_coco_api_from_dataset(test_loader[0].dataset)
    model = MaskRCNN.load_from_checkpoint(
        checkpoint_path="home/ec2-user/gorkem/data-quality-pocs/examples/mask-rcnn/TACO/2xlm62ij/checkpoints/epoch=10-val.ckpt",
        num_classes=len(coco.cats) + 1,
    )

    wandb_logger = (
        WandbLogger(
            entity=params.logging.entity,
            project=params.logging.project,
            log_model=False,
        )
        if params.logging.wandb
        else None
    )

    trainer = pl.Trainer(
        accelerator=params.general.device,
        max_epochs=20,
        logger=wandb_logger,  # type: ignore
        log_every_n_steps=params.logging.log_every_n_steps,
    )

    trainer.test(model=model, dataloaders=test_loader)

    if params.logging.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
