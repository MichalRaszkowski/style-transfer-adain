import torch
from lightning import Trainer

from src.lightning_module import StyleTransferModule
from src.data.data_loader import get_dataloader


def main():
    print("creating dataloader")
    dl = get_dataloader(batch_size=2, max_samples=4)

    print("creating model")
    model = StyleTransferModule()

    batch = next(iter(dl))
    content, style = batch
    print("Content batch:", content.shape)
    print("Style batch:", style.shape)

    print("Forward pass test")
    out, t = model(content, style)
    print("Output:", out.shape)
    print("AdaIN feature:", t.shape)

    print("training_step")
    loss = model.training_step(batch, 0)
    print("Loss OK:", float(loss))

    print("\nworks")


if __name__ == "__main__":
    main()
