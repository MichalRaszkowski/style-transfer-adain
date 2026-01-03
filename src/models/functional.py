def calc_mean_std(feat, eps=1e-5):
    N, C, H, W = feat.shape

    feat_reshaped = feat.view(N, C, H * W)

    mean = feat_reshaped.mean(dim=2)
    var = feat_reshaped.var(dim=2)
    std = (var + eps).sqrt()
    
    mean = mean.view(N, C, 1, 1)
    std = std.view(N, C, 1, 1)

    return mean, std

def adain(content_feat, style_feat):
    style_mean, style_std = calc_mean_std(style_feat)

    content_mean, content_std = calc_mean_std(content_feat)

    normalized_content = (content_feat - content_mean) / content_std

    output = normalized_content * style_std + style_mean
    return output