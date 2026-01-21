import time
import torch
from src.lightning_module import StyleTransferModule

CKPT_PATH = "checkpoints/style-transfer-best-v2.ckpt"
IMG_SIZE = 512
LOOPS = 20

def test_device(model, device_name):
    device = torch.device(device_name)
    model.to(device)
    
    dummy_content = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    dummy_style = torch.rand(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    print(f"Device: {device_name.upper()}")
    
    start = time.time()
    
    with torch.no_grad():
        for i in range(LOOPS):
            _ = model(dummy_content, dummy_style)
            
    total_time = time.time() - start
    
    avg_time = total_time / LOOPS
    fps = 1.0 / avg_time
    
    print(f"\nAvarage time: {avg_time:.4f} s")
    print(f"FPS: {fps:.2f}")

def main():
    print("Loading model")
    model = StyleTransferModule.load_from_checkpoint(CKPT_PATH)
    model.eval()

    test_device(model, "cpu")

    if torch.cuda.is_available():
        test_device(model, "cuda")
    else:
        print("Skipping CUDA test.")

if __name__ == "__main__":
    main()