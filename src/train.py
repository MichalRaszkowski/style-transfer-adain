from src.data_loader import train_dataloader
import torch
from src.models.vgg import get_vgg_feature_extractor

vgg_extractor = get_vgg_feature_extractor()
data_loader = train_dataloader

for content_batch, style_batch in data_loader:
    content_features = vgg_extractor(content_batch)
    style_features = vgg_extractor(style_batch)

    content_feat = content_features['content']['21']
    style_feat = style_features['style']['19']


# with torch.inference_mode():
#     for param in vgg.parameters():
#         param.requires_grad = False

#     for content_batch, style_batch in data_loader:
#         content_features = vgg(content_batch)
#         style_features = vgg(style_batch)
#         print("Content features shape:", content_features.shape)
#         print("Style features shape:", style_features.shape)
#         break  # Just to test one batch
    
