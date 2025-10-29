from torchvision.models import vgg19, VGG19_Weights
import torch.nn as nn

class VGGFeatureExtractor(nn.Module):
    def __init__(self, model, style_layers, content_layers):
        super(VGGFeatureExtractor, self).__init__()
        self.model = model
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.layers = {str(i) : layer for i, layer in enumerate(model)}

    def forward(self, x):
        style_features = {}
        content_features = {}
        for name, layer in self.layers.items():
            x = layer(x)
            if name in self.style_layers:
                style_features[name] = x
            if name in self.content_layers:
                content_features[name] = x
        return {"content": content_features, 'style': style_features}
    
def get_vgg_feature_extractor():
    weights = VGG19_Weights.IMAGENET1K_V1
    vgg = vgg19(weights=weights).features.eval()
    style_layers = ['0', '5', '10', '19', '28']
    content_layers = ['21']
    feature_extractor = VGGFeatureExtractor(vgg, style_layers, content_layers)
    return feature_extractor