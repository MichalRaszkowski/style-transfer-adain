from src.data_loader import train_dataloader
import torch
from src.models.vgg import get_vgg_feature_extractor

vgg_extractor = get_vgg_feature_extractor()
data_loader = train_dataloader


for content_batch, style_batch in data_loader:
    content_features = vgg_extractor(content_batch)
    style_features = vgg_extractor(style_batch)

    content_feat = content_features['content']['0']
    style_feat = style_features['style']['19']




# import matplotlib.pyplot as plt
# import torch

# feat = content_feat[0]
# C, H, W = feat.shape
# print(f"Feature map shape: {feat.shape}")

# channels_to_show = [0, 10, 50, 63]
# plt.figure(figsize=(15,3))
# for i, ch in enumerate(channels_to_show):
#     ax = plt.subplot(1, len(channels_to_show), i+1)
#     channel = feat[ch].detach().cpu()
#     channel = (channel - channel.min()) / (channel.max() - channel.min())
#     ax.imshow(channel, cmap='viridis')
#     ax.axis('off')
#     ax.set_title(f'Channel {ch}')
# plt.show()

# mean_activation = feat.mean(dim=0)
# mean_activation = (mean_activation - mean_activation.min()) / (mean_activation.max() - mean_activation.min())

# plt.figure(figsize=(4,4))
# plt.imshow(mean_activation.detach().cpu(), cmap='viridis')
# plt.title('Average activation (content features)')
# plt.axis('off')
# plt.show()


# with torch.inference_mode():
#     for param in vgg.parameters():
#         param.requires_grad = False

#     for content_batch, style_batch in data_loader:
#         content_features = vgg(content_batch)
#         style_features = vgg(style_batch)
#         print("Content features shape:", content_features.shape)
#         print("Style features shape:", style_features.shape)
#         break  # Just to test one batch
    
