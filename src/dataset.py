import random
from torch.utils.data import Dataset
from torchvision import transforms

class DecoderDataset(Dataset):
    def __init__(
        self,
        content_dataset,
        style_dataset,
        content_indexes,
        style_indexes,
        image_size=256,
        is_train=True
    ):

        self.content_dataset = content_dataset
        self.style_dataset = style_dataset
        self.content_indexes = content_indexes
        self.style_indexes = style_indexes
        self.is_train = is_train
        
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.content_indexes)

    def __getitem__(self, idx):
        real_content_idx = self.content_indexes[idx]
        
        if self.is_train:
            style_idx = random.choice(self.style_indexes)
        else:
            style_idx = self.style_indexes[idx % len(self.style_indexes)]

        content_data = self.content_dataset[real_content_idx]
        style_data = self.style_dataset[style_idx]
            
        content_img = content_data["image"]
        style_img = style_data["image"]

        return self.transform(content_img.convert("RGB")), self.transform(style_img.convert("RGB"))