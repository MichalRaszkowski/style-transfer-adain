import gradio as gr
import torch
import os
from torchvision import transforms
from PIL import Image
from src.lightning_module import StyleTransferModule


MODEL_URL = "https://huggingface.co/Michal-Raszkowski/adain-style-transfer/resolve/main/style-transfer-best-v2.ckpt?download=true"
CHECKPOINT_PATH = "model.ckpt"

def download_model_if_missing():
    if not os.path.exists(CHECKPOINT_PATH):
        torch.hub.download_url_to_file(MODEL_URL, CHECKPOINT_PATH)

def load_model():
    download_model_if_missing()
    model = StyleTransferModule.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
    model.eval()
    return model

model = load_model()

def stylize(content_image, style_image, alpha):
    if content_image is None or style_image is None:
        return None
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    c = transform(content_image).unsqueeze(0)
    s = transform(style_image).unsqueeze(0)
    
    with torch.no_grad():
        generated_tensor, _ = model(c, s, alpha=alpha)
    
    generated_tensor = torch.clamp(generated_tensor, 0, 1)
    result_image = transforms.ToPILImage()(generated_tensor.squeeze(0))
    
    return result_image

with gr.Blocks(title="Style Transfer Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("Neural Style Transfer")
    gr.Markdown("Upload content and style images to combine them.")
    
    with gr.Row():
        with gr.Column():
            input_content = gr.Image(label="Content image", type="pil", height=300)
            input_style = gr.Image(label="Style image", type="pil", height=300)
            slider = gr.Slider(minimum=0.0, maximum=1.0, value=1.0, label="Style strenght.")
            btn = gr.Button("Generate", variant="primary")
        
        with gr.Column():
            output = gr.Image(label="Output", type="pil")
    
    btn.click(fn=stylize, inputs=[input_content, input_style, slider], outputs=output)
    
    #gr.Examples(examples=[["examples/c.jpg", "examples/s.jpg", 1.0]], inputs=[input_content, input_style, slider])

if __name__ == "__main__":
    demo.launch()