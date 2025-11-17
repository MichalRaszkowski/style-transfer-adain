import torch
from torchvision.utils import save_image
from src.models.vgg import get_vgg_feature_extractor
from src.models.decoder import Decoder
from src.models.adain import adain
from src.data.data_loader import train_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = get_vgg_feature_extractor().to(device).eval()
decoder = Decoder().to(device)
decoder.load_state_dict(torch.load("decoder.pth", map_location=device))
decoder.eval()

content_batch, style_batch = next(iter(train_dataloader))
content_batch, style_batch = content_batch.to(device), style_batch.to(device)

content_features = vgg(content_batch)
style_features = vgg(style_batch)

content_layer = '21'
style_layer = '19'
t = adain(
    content_features['content'][content_layer],
    style_features['style'][style_layer]
)

with torch.no_grad():
    output = decoder(t)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

content_vis = denormalize(content_batch)
style_vis = denormalize(style_batch)
output_vis = denormalize(output)

save_image(content_vis, "content_example.jpg")
save_image(style_vis, "style_example.jpg")
save_image(output_vis, "stylized_example.jpg")

print("Imges saved: content_example.jpg, style_example.jpg, stylized_example.jpg")
