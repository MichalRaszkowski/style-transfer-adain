import torch
import torch.nn as nn
import lightning as L

from src.models.vgg import get_vgg_feature_extractor
from src.models.decoder import Decoder
from src.models.adain import adain


class StyleTransferModule(L.LightningModule):
    def __init__(
        self,
        content_weight=1.0,
        style_weight=10.0,
        learning_rate=1e-4
    ):
        super().__init__()

        self.save_hyperparameters()

        self.vgg = get_vgg_feature_extractor()
        self.decoder = Decoder()

        self.mse = nn.MSELoss()

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, content_img, style_img):
        c_feats = self.vgg(content_img)["content"]
        s_feats = self.vgg(style_img)["style"]

        c_feat = c_feats["21"]
        s_feat = s_feats["28"]

        t = adain(c_feat, s_feat)

        out = self.decoder(t)

        return out, t

    def training_step(self, batch, batch_idx):
        content_img, style_img = batch

        generated_img, t = self(content_img, style_img)

        gen_feats = self.vgg(generated_img)
        c_feats = self.vgg(content_img)
        s_feats = self.vgg(style_img)

        content_loss = self.mse(
            gen_feats["content"]["21"],
            t
        )

        style_loss = 0.0
        for layer in s_feats["style"]:
            gen_f = gen_feats["style"][layer]
            style_f = s_feats["style"][layer]

            style_loss += self.mse(
                gen_f.mean([2, 3]),
                style_f.mean([2, 3])
            ) + self.mse(
                gen_f.std([2, 3]),
                style_f.std([2, 3])
            )

        loss = (
            self.hparams.content_weight * content_loss +
            self.hparams.style_weight * style_loss
        )

        self.log("train_loss", loss)
        self.log("content_loss", content_loss)
        self.log("style_loss", style_loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.decoder.parameters(),
            lr=self.hparams.learning_rate
        )
