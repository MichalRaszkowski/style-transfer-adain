from torchvision.models import vgg19, VGG19_Weights
import torch
from src.data_loader import train_dataloader

weights = VGG19_Weights.IMAGENET1K_V1
vgg = vgg19(weights=weights).features.eval()

data_loader = train_dataloader

with torch.inference_mode():
    for param in vgg.parameters():
        param.requires_grad = False

    for content_batch, style_batch in data_loader:
        content_features = vgg(content_batch)
        style_features = vgg(style_batch)
        print("Content features shape:", content_features.shape)
        print("Style features shape:", style_features.shape)
        break  # Just to test one batch
    