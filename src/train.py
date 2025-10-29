import torch
import torch.nn.functional as F
from src.models.adain import adain
from src.models.decoder import Decoder
from src.models.vgg import get_vgg_feature_extractor
from src.data_loader import train_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg = get_vgg_feature_extractor().to(device).eval()
decoder = Decoder().to(device)
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)

style_layers = ['0', '5', '10', '19', '28']
content_layer = '21'
style_layer = '19'

for epoch in range(5):
    for content_batch, style_batch in train_dataloader:
        content_batch = content_batch.to(device)
        style_batch = style_batch.to(device)

        content_features = vgg(content_batch)
        with torch.no_grad():
            style_features = vgg(style_batch)

        t = adain(
            content_features['content'][content_layer],
            style_features['style'][style_layer]
        )

        output = decoder(t)

        output_features = vgg(output)

        loss_c = F.mse_loss(
            output_features['content'][content_layer],
            t
        )

        loss_s = 0
        for layer in style_layers:
            out_feat = output_features['style'][layer]
            style_feat = style_features['style'][layer]

            mu_out, sigma_out = out_feat.mean([2,3]), out_feat.std([2,3])
            mu_style, sigma_style = style_feat.mean([2,3]), style_feat.std([2,3])

            loss_s += F.mse_loss(mu_out, mu_style) + F.mse_loss(sigma_out, sigma_style)

        loss = loss_c + 10.0 * loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
