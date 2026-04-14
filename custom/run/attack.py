import torch
import os
import torchvision.utils as vutils
import torchvision.transforms as transforms

from custom.component.gradient_inversion_attack import GradientInversionAttack
from custom.model.grad_mlp import model_build_fn
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset

# ========================
# CONFIG
# ========================
MODEL_PATH = "results/dual_dp_n12_r50_e1_bs32_clip_norm10000_delta1e-05_scale_modestandard_alpha0.5_beta0.5_gamma1.0_k5_epsilon_base50_epsilon_min20_lambda_protection1_nu1/final_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ATTACK_DIR = "attack_results"
NUM_SAMPLES = 50


def main():
    print("🚨 Running Gradient Inversion Attack (simple)...")
    
    config_name = os.path.basename(os.path.dirname(MODEL_PATH))

    # tạo save dir
    ATTACK_SAVE_DIR = os.path.join(ATTACK_DIR, config_name)
    os.makedirs(ATTACK_SAVE_DIR, exist_ok=True)
    print("📁 Attack save dir:", ATTACK_SAVE_DIR)

    # ========================
    # LOAD MODEL
    # ========================
    assert os.path.exists(MODEL_PATH), "❌ Model path not found!"

    model = model_build_fn().model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    print("✅ Model loaded")

    # ========================
    # LOAD DATA (1 sample)
    # ========================

    data = P2PFLDataset.from_huggingface("p2pfl/MNIST")

    transform = transforms.ToTensor()

    data_list = []
    
    attack: GradientInversionAttack = GradientInversionAttack(
        method="iDLG",
        iters=5,
        lr=1,
        device=DEVICE,
        save_dir=ATTACK_SAVE_DIR
    )

    os.makedirs(ATTACK_SAVE_DIR, exist_ok=True)
    

    # ========================
    # ATTACK
    # ========================
    
    for i in range(NUM_SAMPLES):
        sample = data.get(i)

        x = transform(sample["image"]).squeeze(0).unsqueeze(0)
        y = torch.tensor([sample["label"]])

        data_list.append((x, y))

    attack.reconstruct_multiple(
        model=model,
        data_list=data_list,
        num_classes=10
    )

    # ========================
    # EVALUATE
    # ========================

    print(f"✅ Saved in {ATTACK_SAVE_DIR}")


if __name__ == "__main__":
    main()