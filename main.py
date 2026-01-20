import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.lightning_module import StyleTransferModule
from src.data_module import StyleTransferDataModule
from src.callbacks import ImageLogger

def main():
    BATCH_SIZE = 64
    LR = 1e-4
    MAX_EPOCHS = 300
    
    dm = StyleTransferDataModule(
        batch_size=BATCH_SIZE,
        image_size=256,
        val_split=0.05, 
        test_split=0.05,
        max_samples=50000 
    )

    model = StyleTransferModule(learning_rate=LR, content_weight=1.0, style_weight=10.0)

    wandb_logger = WandbLogger(project="ADAIN-Style-Transfer", name="run-v5")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/v5",
        filename="{epoch}-{val/loss:.2f}",
        every_n_epochs=1,
        save_top_k=3,
        monitor="val/loss",
        mode="min"
    )

    img_logger = ImageLogger(num_samples=8)

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        logger=wandb_logger,
        callbacks=[checkpoint_callback, img_logger],
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )

    trainer.fit(model, datamodule=dm)

    #print("Testing on test set: ")
    #trainer.test(model, datamodule=dm, ckpt_path="checkpoints/v4/epoch=264-val/loss=40.02.ckpt")

if __name__ == "__main__":
    main()