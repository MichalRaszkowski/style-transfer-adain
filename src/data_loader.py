import torch
from torch.utils.data import DataLoader
from src.dataset import DecoderDataset

# change it later to bigger dataset
content_dataset_name = "CongWei1230/objectswap_coco_s0_100_samples"

style_dataset_name = "newsletter/HiDream-I1-Artists"


train_dataset = DecoderDataset(
    content_dataset_name=content_dataset_name,
    style_dataset_name=style_dataset_name,
    content_split="train",
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