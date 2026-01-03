import torch
import os
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from src.lightning_module import StyleTransferModule

def load_image(path, size):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path not found")
        
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style", type=str, required=True)
    
    parser.add_argument("--output", type=str, default="output.jpg")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=512)
    
    return parser.parse_args()

def main():
    args = get_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.ckpt}")
    
    try:
        model = StyleTransferModule.load_from_checkpoint("checkpoints/style-transfer-best.ckpt", map_location=device)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Couldn't load the model.\n{e}")
        return

    try:
        content = load_image(args.content, args.size).to(device)
        style = load_image(args.style, args.size).to(device)
    except FileNotFoundError as e:
        print(f"Coudn't load images: {e}")
        return

    with torch.no_grad():
        generated_tensor, _ = model(content, style, alpha=args.alpha)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    save_image(generated_tensor, args.output)
    print(f"Image saved in: {args.output}")

if __name__ == "__main__":
    main()