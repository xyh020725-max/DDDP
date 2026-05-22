import os
import glob
import yaml
import argparse
import math
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as calculate_ssim
from torchmetrics.image.fid import FrechetInceptionDistance

from guided_diffusion.icsa_net import ICSA_Net

class EvalDataset(Dataset):
    def __init__(self, img_dir, max_imgs=None):
        self.paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ])
        if max_imgs is not None: self.paths = self.paths[:max_imgs]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))

def unnormalize(x):
    return torch.clamp((x + 1) / 2, 0, 1)

# ---------------------------------------------------------
# 固定且具有挑战性的盲去模糊 Benchmark 退化
# ---------------------------------------------------------
def apply_benchmark_blur(x, device):
    # 使用统一的 7x7 核，Sigma=2.0 模拟极其严重的失焦
    kernel_size = 7
    sigma = 2.0
    blur = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    return blur

# ---------------------------------------------------------
# 纯神经网络生成 (无任何物理公式作弊)
# ---------------------------------------------------------
@torch.inference_mode()
def ddpm_blind_sample(model, Apy, config, device):
    B, C, H, W = Apy.shape
    xt = torch.randn((B, C, H, W), device=device)

    T = config.diffusion.num_diffusion_timesteps
    betas = np.linspace(0.0001, 0.02, T, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    betas = torch.tensor(betas, device=device).float()
    alphas = torch.tensor(alphas, device=device).float()
    alpha_bar = torch.tensor(alpha_bar, device=device).float()

    for t in tqdm(reversed(range(0, T)), desc="Blind Deblurring", leave=False):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        # 仅靠模型自己去预测噪声，没有任何 SVD 校准！
        eps = model(xt, t_batch, Apy=Apy)

        a = alpha_bar[t]
        x0_pred = (xt - torch.sqrt(1 - a) * eps) / torch.sqrt(a)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        posterior_mean = (torch.sqrt(alpha_bar_prev) * beta_t * x0_pred +
                          torch.sqrt(alpha_t) * (1 - alpha_bar_prev) * xt) / (1 - a)

        if t > 0:
            noise = torch.randn_like(xt)
            xt = posterior_mean + torch.sqrt(beta_t) * noise
        else:
            xt = posterior_mean

    return torch.clamp(xt, -1.0, 1.0)

def psnr_standard(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0: return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_metrics(pred_uint8, gt_uint8):
    pred_np = pred_uint8.cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt_uint8.cpu().numpy().transpose(0, 2, 3, 1)
    B = pred_uint8.shape[0]
    ps, ss = 0.0, 0.0
    for i in range(B):
        ps += psnr_standard(pred_np[i], gt_np[i])
        ss += calculate_ssim(
            gt_np[i], pred_np[i], data_range=255, channel_axis=-1,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False
        )
    return ps / B, ss / B

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="celeba_hq.yml")
    parser.add_argument("--val_dir", default="exp/datasets/celeba_hq_standard/val/face")
    parser.add_argument("--ckpt_dir", default="exp/logs/icsa_deblur") # 指向专属文件夹
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_imgs", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join("configs", args.config)) as f: cfg = yaml.safe_load(f)
    class Struct:
        def __init__(self, **entries): self.__dict__.update(entries)
    config = Struct(**{k: Struct(**v) if isinstance(v, dict) else v for k, v in cfg.items()})

    dataset = EvalDataset(args.val_dir, max_imgs=args.num_imgs)
    loader = DataLoader(dataset, args.batch_size, shuffle=False)

    model = ICSA_Net(config).to(device)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    if not ckpts: 
        print("未找到专属 Deblur 权重文件！")
        return
    latest_ckpt = ckpts[-1]
    name = os.path.basename(latest_ckpt)
    
    print(f"\n🚀 正在进行 Blind Deblur 极限评估: {name} (共 {len(dataset)} 张图)")
    state = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'] if "model_state_dict" in state else state)
    model.eval()

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    total_ps, total_ss, total_batches = 0.0, 0.0, 0

    for step, gt in enumerate(tqdm(loader)):
        gt = gt.to(device)
        
        # 获取基准模糊图像
        Apy = apply_benchmark_blur(gt, device)

        # 纯网络推理
        out = ddpm_blind_sample(model, Apy, config, device)

        out_01 = unnormalize(out)
        gt_01 = unnormalize(gt)
        Apy_01 = unnormalize(Apy)
        
        out_uint8 = (out_01 * 255.0).clamp(0, 255).to(torch.uint8)
        gt_uint8 = (gt_01 * 255.0).clamp(0, 255).to(torch.uint8)

        fid_metric.update(gt_uint8, real=True)
        fid_metric.update(out_uint8, real=False)

        p, s = compute_metrics(out_uint8, gt_uint8)
        total_ps += p
        total_ss += s
        total_batches += 1

        if step == 0:
            vis = torch.cat([gt_01, Apy_01, out_01], dim=0)
            os.makedirs("eval_vis", exist_ok=True)
            vutils.save_image(vis, f"eval_vis/blind_deblur_example.png", nrow=3)

    avg_ps = total_ps / total_batches
    avg_ss = total_ss / total_batches
    fid_score = fid_metric.compute().item()

    print("\n" + "=" * 60)
    print(f"🏆 {name} - Blind Deblur 挑战报告 🏆")
    print("-" * 60)
    print(f"{'Task':<15} | {'PSNR ↑':<10} | {'SSIM ↑':<10} | {'FID ↓':<10}")
    print("-" * 60)
    print(f"{'BLIND DEBLUR':<15} | {avg_ps:>6.2f}     | {avg_ss:>6.4f}     | {fid_score:>6.2f}")
    print("=" * 60)
    print("📸 极度瞎眼盲盒重建成图已保存至 eval_vis/blind_deblur_example.png！")

if __name__ == "__main__":
    main()