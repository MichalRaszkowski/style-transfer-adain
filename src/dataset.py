from torchvision import transforms
from datasets import load_dataset, IterableDataset


class DecoderDataset(IterableDataset):
    def __init__(
        self,
        content_dataset_name: str,
        style_dataset_name: str,
        content_split="train",
        style_split="train",
        image_size=256,
        streaming=True,
        max_samples=None
    ):
        self.content_dataset = load_dataset(
            content_dataset_name,
            split=content_split,
            streaming=streaming
        )
        self.style_dataset = load_dataset(
            style_dataset_name,
            split=style_split,
            streaming=streaming
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.max_samples = max_samples

    def __iter__(self):
        for i, (content_image, style_image) in enumerate(
            zip(self.content_dataset, self.style_dataset)
        ):
            if self.max_samples and i >= self.max_samples:
                break

            content_image = self.transform(content_image["image"])
            style_image = self.transform(style_image["image"])

            yield content_image, style_image
