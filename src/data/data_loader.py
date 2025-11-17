import torch
from torch.utils.data import DataLoader
from src.dataset import DecoderDataset


def get_dataloader(
    content_dataset_name="wangwangxuebing/train_COCO_data2014_jpg",
    style_dataset_name="newsletter/HiDream-I1-Artists",
    content_split="test",
    style_split="train",
    batch_size=4,
    image_size=256,
    streaming=True,
    max_samples=100,
    num_workers=0,
    shuffle=False
):
    dataset = DecoderDataset(
        content_dataset_name=content_dataset_name,
        style_dataset_name=style_dataset_name,
        content_split=content_split,
        style_split=style_split,
        image_size=image_size,
        streaming=streaming,
        max_samples=max_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return dataloader
