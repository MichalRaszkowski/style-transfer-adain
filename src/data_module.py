import lightning as L
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from src.dataset import DecoderDataset

class StyleTransferDataModule(L.LightningDataModule):
    def __init__(self, batch_size=8, image_size=256, val_split=0.05, test_split=0.05, max_samples=None):
        super().__init__()
        self.save_hyperparameters()
        
        self.content_name = "wangwangxuebing/train_COCO_data2014_jpg"
        self.style_name = "huggan/wikiart"

    def setup(self, stage=None):
        full_content = load_dataset(self.content_name, split="test")
        full_style = load_dataset(self.style_name, split="train")

        if self.hparams.max_samples:
            full_content = full_content.select(range(min(len(full_content), self.hparams.max_samples)))
            full_style = full_style.select(range(min(len(full_style), self.hparams.max_samples)))

        seed = 42
        content_indexes = np.random.RandomState(seed).permutation(len(full_content))
        style_indexes = np.random.RandomState(seed).permutation(len(full_style))

        def get_splits(all_indexes, val_ratio, test_ratio):
            total_count = len(all_indexes)
            
            n_val = int(total_count * val_ratio)
            n_test = int(total_count * test_ratio)
            n_train = total_count - n_val - n_test

            train_end = n_train
            val_end = n_train + n_val

            train_indices = all_indexes[:train_end]
            val_indices   = all_indexes[train_end:val_end]
            test_indices  = all_indexes[val_end:]
            
            return train_indices, val_indices, test_indices

        content_train, content_val, content_test = get_splits(content_indexes, self.hparams.val_split, self.hparams.test_split)
        style_train, style_val, style_test = get_splits(style_indexes, self.hparams.val_split, self.hparams.test_split)

        if stage == "fit" or stage is None:
            self.train_ds = DecoderDataset(full_content, full_style, content_train, style_train, self.hparams.image_size, is_train=True)
            self.val_ds = DecoderDataset(full_content, full_style, content_val, style_val, self.hparams.image_size, is_train=False)

        if stage == "test":
            self.test_ds = DecoderDataset(full_content, full_style, content_test, style_test, self.hparams.image_size, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=12, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=4)