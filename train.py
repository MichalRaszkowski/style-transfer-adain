import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from src.data.data_loader import get_dataloader
from src.lightning_module import StyleTransferModule


def main():
    # -------- DATA --------
    train_loader = get_dataloader(
        content_dir="data/content",
        style_dir="data/style",
        batch_size=4,
        img_size=256,
        shuffle=True
    )

    # -------- MODEL --------
    model = StyleTransferModule(lr=1e-4)

    # -------- CALLBACKI --------
    checkpoint = ModelCheckpoint(
        save_top_k=-1,        # zapisuje wszystkie epochy
        every_n_epochs=1,
        dirpath="checkpoints",
        filename="epoch_{epoch}"
    )

    logger = CSVLogger("logs", name="style-transfer")

    # -------- TRAINER --------
    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",  # użyje GPU jeśli dostępne
        devices="auto",
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=1
    )

    # -------- START --------
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
