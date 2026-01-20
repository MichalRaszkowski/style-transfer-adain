import torch
import torch.nn as nn
import lightning as L
from src.models import VGGEncoder, Decoder, adain, calc_mean_std
from torchmetrics.image import StructuralSimilarityIndexMeasure

class StyleTransferModule(L.LightningModule):
    def __init__(self, content_weight=1.0, style_weight=10.0, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VGGEncoder()
        self.decoder = Decoder()
        self.mse_loss = nn.MSELoss()

        self.register_buffer('vgg_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('vgg_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def normalize_vgg(self, x):
        return (x - self.vgg_mean) / self.vgg_std

    def forward(self, content_img, style_img, alpha=1.0):
        content_feats = self.encoder(self.normalize_vgg(content_img))
        style_feats = self.encoder(self.normalize_vgg(style_img))
        
        content_feat = content_feats[3]
        style_feat = style_feats[3]
        
        target = adain(content_feat, style_feat)
        
        if alpha < 1.0:
            target = alpha * target + (1 - alpha) * content_feat
            
        generated_img = self.decoder(target)
        return generated_img, target

    def calculate_loss(self, content_img, style_img):
        generated_img, target = self(content_img, style_img)
        
        g_img_norm = self.normalize_vgg(generated_img)
        g_feats = self.encoder(g_img_norm)
        
        content_loss = self.mse_loss(g_feats[3], target)
        
        s_feats = self.encoder(self.normalize_vgg(style_img))
        style_loss = 0.0
        for g_f, s_f in zip(g_feats, s_feats):
            g_mean, g_std = calc_mean_std(g_f)
            s_mean, s_std = calc_mean_std(s_f)
            style_loss += self.mse_loss(g_mean, s_mean) + self.mse_loss(g_std, s_std)

        total_loss = (self.hparams.content_weight * content_loss) + \
                     (self.hparams.style_weight * style_loss)
                     
        return total_loss, content_loss, style_loss

    def training_step(self, batch, batch_idx):
        loss, c_loss, s_loss = self.calculate_loss(*batch)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/content", c_loss)
        self.log("train/style", s_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        c, s = batch
        generated_img, _ = self(c, s)
        ssim_score = self.metric_ssim(
            torch.clamp(generated_img, 0, 1), 
            torch.clamp(c, 0, 1)
        )
        self.log("val/ssim", ssim_score, on_step=False, on_epoch=True, prog_bar=True)
        
        loss, c_loss, s_loss = self.calculate_loss(*batch)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/content", c_loss)
        self.log("val/style", s_loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, c_loss, s_loss = self.calculate_loss(*batch)
        self.log("test/loss", loss)
        self.log("test/content", c_loss)
        self.log("test/style", s_loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.decoder.parameters(), lr=self.hparams.learning_rate)