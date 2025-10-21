import torch

def adain(content_features: torch.Tensor, style_features: torch.Tensor, eps=1e-5) -> torch.Tensor:

    c_mean = torch.mean(content_features, dim=[2, 3], keepdim=True)
    c_std  = torch.std(content_features, dim=[2, 3], keepdim=True) + eps
    s_mean = torch.mean(style_features, dim=[2, 3], keepdim=True)
    s_std  = torch.std(style_features, dim=[2, 3], keepdim=True) + eps

    normalized_content = (content_features - c_mean) / c_std

    stylized_features = normalized_content * s_std + s_mean

    return stylized_features