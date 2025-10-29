import torch
from torch.utils.data import DataLoader
from src.dataset import DecoderDataset

content_dataset_name = "wangwangxuebing/train_COCO_data2014_jpg"

style_dataset_name = "newsletter/HiDream-I1-Artists"


train_dataset = DecoderDataset(
    content_dataset_name=content_dataset_name,
    style_dataset_name=style_dataset_name,
    content_split="test",
    style_split="train",
    image_size=256,
    streaming=True,
    max_samples=100
    )

train_dataloader = DataLoader(train_dataset, batch_size=4, num_workers=0)

for content_batch, style_batch in train_dataloader:
    print("Content batch shape:", content_batch.shape)
    print("Style batch shape:", style_batch.shape)
    break  # Just to test one batch

