import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from model import CNN

def export_float32_model():
    
    # Load CNN & weights
    model = CNN(input_size=50, num_classes=3)
    model.load_state_dict(
        torch.load("export/weights/best_model.pth", map_location="cpu")
    )
    model.eval()

    # Dummy input [1,1,50,50]
    example = torch.randn(1, 1, 50, 50)

    # Trace & optimize
    traced = torch.jit.trace(model, example)
    optimized = optimize_for_mobile(traced)

    # Save for Lite interpreter
    out_path = "export/float32/rpsmodel.ptl"
    optimized._save_for_lite_interpreter(out_path)
    print(f"Float32 lite model saved at {out_path}")

if __name__ == "__main__":
    export_float32_model()