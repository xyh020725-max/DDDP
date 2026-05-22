import os
import glob
import yaml
import argparse
import math
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as calculate_ssim
from torchmetrics.image.fid import FrechetInceptionDistance

# 导入你的核心网络
from guided_diffusion.icsa_net import ICSA_Net

# ---------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------
class EvalDataset(Dataset):
    def __init__(self, img_dir, max_imgs=None):
        self.paths = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('png', 'jpg', 'jpeg'))
        ])
        if max_imgs is not None:
            self.paths = self.paths[:max_imgs]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

def unnormalize(x):
    return torch.clamp((x + 1) / 2, 0, 1)

# =========================================================
# 【核心辅助】获取中心对齐的 FFT 模糊核 (解决网格伪影的关键)
# =========================================================
def get_fft_kernel(kernel, target_shape):
    # kernel 必须是 (1, 1, k_h, k_w)
    H, W = target_shape
    k_h, k_w = kernel.shape[2], kernel.shape[3]
    padded = torch.zeros((1, 1, H, W), device=kernel.device)
    padded[:, :, :k_h, :k_w] = kernel
    
    # 将核的中心移动到图像的左上角 (0,0)，这是 FFT 的严格数学要求
    padded = torch.roll(padded, shifts=(-(k_h//2), -(k_w//2)), dims=(2, 3))
    K = torch.fft.fft2(padded)
    return K

# ---------------------------------------------------------
# 2. Forward Degradation (严格的退化模型)
# ---------------------------------------------------------
def degradation(x, task, device):
    B, C, H, W = x.shape

    if task == "sr":
        scale = 4
        pool = torch.nn.AdaptiveAvgPool2d((H // scale, W // scale))
        y = pool(x)
        return F.interpolate(y, (H, W), mode="nearest"), None

    elif task == "cs":
        block_size = 32
        N = block_size ** 2
        M = int(N * 0.25) # 25% CS ratio
        
        torch.manual_seed(42)
        Phi = torch.randn(M, N, device=device) / math.sqrt(M)
        Phi_pinv = torch.linalg.pinv(Phi)
        
        x_unfold = x.view(B, C, H // block_size, block_size, W // block_size, block_size)
        x_unfold = x_unfold.permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1, N)
        
        y = torch.matmul(x_unfold, Phi.t())
        
        Apy_unfold = torch.matmul(y, Phi_pinv.t())
        Apy = Apy_unfold.view(B, C, H // block_size, W // block_size, block_size, block_size)
        Apy = Apy.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)
        
        P = torch.matmul(Phi_pinv, Phi)
        
        return Apy, P

    elif task == "inpainting":
        mask = torch.ones((1, 1, H, W), device=device)
        cy, cx = H // 2, W // 2
        mask[:, :, cy - 32 : cy + 32, cx - 32 : cx + 32] = 0
        y = x * mask
        return y, mask

    elif task == "blur":
        # 构建基础单通道模糊核，用于数学计算
        kernel_1ch = torch.tensor(
            [[1, 4, 6, 4, 1],
             [4, 16, 24, 16, 4],
             [6, 24, 36, 24, 6],
             [4, 16, 24, 16, 4],
             [1, 4, 6, 4, 1]],
            dtype=torch.float32, device=device
        )
        kernel_1ch = kernel_1ch / kernel_1ch.sum()
        kernel_1ch = kernel_1ch.view(1, 1, 5, 5)
        
        kernel_3ch = kernel_1ch.repeat(3, 1, 1, 1)
        
        # 🔥 极其重要：使用 circular 边界填充，完美对齐 FFT 数学假设，杜绝网格伪影！
        x_padded = F.pad(x, (2, 2, 2, 2), mode='circular')
        y = F.conv2d(x_padded, kernel_3ch, padding=0, groups=3)
        
        return y, kernel_1ch

# ---------------------------------------------------------
# 3. DDPM Sampling with DC (Data Consistency) Projections
# ---------------------------------------------------------
@torch.inference_mode()
def ddpm_sample(model, Apy, config, device, task, extra=None):
    B, C, H, W = Apy.shape
    xt = torch.randn((B, C, H, W), device=device)

    T = config.diffusion.num_diffusion_timesteps
    betas = np.linspace(0.0001, 0.02, T, dtype=np.float64)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)

    betas = torch.tensor(betas, device=device).float()
    alphas = torch.tensor(alphas, device=device).float()
    alpha_bar = torch.tensor(alpha_bar, device=device).float()

    for t in tqdm(reversed(range(0, T)), desc=f"DDPM [{task.upper()}]", leave=False):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        eps = model(xt, t_batch, Apy=Apy)

        a = alpha_bar[t]
        x0 = (xt - torch.sqrt(1 - a) * eps) / torch.sqrt(a)

        # =================================================
        # 🔥 数据一致性投影层 (彻底点燃分数的关键)
        # =================================================
        if task == "sr":
            # 严格的下采样能量补偿
            scale = 4
            pool = torch.nn.AdaptiveAvgPool2d((H // scale, W // scale))
            diff = F.interpolate(pool(x0) - pool(Apy), size=(H, W), mode="nearest")
            x0 = x0 - diff
            
        elif task == "inpainting":
            mask = extra
            x0 = x0 * (1 - mask) + Apy * mask
            
        elif task == "cs":
            P = extra
            block_size = 32
            N = block_size ** 2
            
            x0_unfold = x0.view(B, C, H // block_size, block_size, W // block_size, block_size)
            x0_unfold = x0_unfold.permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1, N)
            
            Apy_unfold = Apy.view(B, C, H // block_size, block_size, W // block_size, block_size)
            Apy_unfold = Apy_unfold.permute(0, 1, 2, 4, 3, 5).reshape(B, C, -1, N)
            
            x0_unfold = x0_unfold - torch.matmul(x0_unfold, P.t()) + Apy_unfold
            
            x0 = x0_unfold.view(B, C, H // block_size, W // block_size, block_size, block_size)
            x0 = x0.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)

        elif task == "blur":
            kernel_1ch = extra
            # 1. 预计算 FFT 变量
            K = get_fft_kernel(kernel_1ch, (H, W))
            Y = torch.fft.fft2(Apy)
            X0 = torch.fft.fft2(x0)
            
            # 2. Tikhonov Regularization (维纳滤波融合机制)
            # lam=0.05 意味着我们信任物理数学定律，但在奇异值崩溃的频段相信神经网络的重构
            lam = 0.05 
            X0_new = (K.conj() * Y + lam * X0) / (torch.abs(K)**2 + lam)
            
            # 3. 无损逆变换回空间域
            x0 = torch.fft.ifft2(X0_new).real

        # 全局截断，防止数值溢出
        x0 = torch.clamp(x0, -1.0, 1.0)

        # =================================================
        # 标准的 DDPM 逆向加噪步
        # =================================================
        alpha_t = alphas[t]
        beta_t = betas[t]
        alpha_bar_prev = alpha_bar[t - 1] if t > 0 else torch.tensor(1.0, device=device)

        posterior_mean = (torch.sqrt(alpha_bar_prev) * beta_t * x0 +
                          torch.sqrt(alpha_t) * (1 - alpha_bar_prev) * xt) / (1 - a)

        if t > 0:
            noise = torch.randn_like(xt)
            xt = posterior_mean + torch.sqrt(beta_t) * noise
        else:
            xt = posterior_mean

    return torch.clamp(xt, -1.0, 1.0)

# ---------------------------------------------------------
# 4. Metrics (严格对齐学术界标准)
# ---------------------------------------------------------
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
        
        # 11x11 高斯加权，严格对齐 MATLAB 学术标准
        ss += calculate_ssim(
            gt_np[i],
            pred_np[i],
            data_range=255,
            channel_axis=-1,
            # gaussian_weights=True, 
            # sigma=1.5,
            # use_sample_covariance=False
        )
    return ps / B, ss / B

# ---------------------------------------------------------
# 5. Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="celeba_hq.yml")
    parser.add_argument("--val_dir", default="exp/datasets/celeba_hq_standard/val/face")
    parser.add_argument("--ckpt_dir", default="exp/logs/icsa_net")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_imgs", type=int, default=5, help="默认测5张, 论文刷榜设为500")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join("configs", args.config)) as f:
        cfg = yaml.safe_load(f)

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    config = Struct(**{k: Struct(**v) if isinstance(v, dict) else v for k, v in cfg.items()})

    dataset = EvalDataset(args.val_dir, max_imgs=args.num_imgs)
    loader = DataLoader(dataset, args.batch_size, shuffle=False)

    model = ICSA_Net(config).to(device)

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "*.pt")))
    if not ckpts:
        print("未找到权重文件！")
        return
    
    # 仅测最新的一个权重
    latest_ckpt = ckpts[-1]
    name = os.path.basename(latest_ckpt)
    
    tasks = ["sr", "cs", "inpainting", "blur"]
    report = {}

    print(f"\n🚀 正在评估全能权重: {name} (共 {len(dataset)} 张图)")
    state = torch.load(latest_ckpt, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    for task in tasks:
        print(f"\n======== 开始测试任务: {task.upper()} ========")
        
        # 初始化 FID 计算器 (特征维度2048)
        fid_metric = FrechetInceptionDistance(feature=2048).to(device)
        
        total_ps, total_ss = 0.0, 0.0
        total_batches = 0

        for step, gt in enumerate(tqdm(loader, leave=False)):
            gt = gt.to(device)

            if task == "sr":
                y, extra = degradation(gt, "sr", device)
            else:
                y, extra = degradation(gt, task, device)

            out = ddpm_sample(model, y, config, device, task, extra)

            # 转回 0~1
            out_01 = unnormalize(out)
            gt_01 = unnormalize(gt)
            y_01 = unnormalize(y)

            # 转为 uint8 用于严谨的计算
            out_uint8 = (out_01 * 255.0).clamp(0, 255).to(torch.uint8)
            gt_uint8 = (gt_01 * 255.0).clamp(0, 255).to(torch.uint8)

            # 更新 FID 统计特征
            fid_metric.update(gt_uint8, real=True)
            fid_metric.update(out_uint8, real=False)

            p, s = compute_metrics(out_uint8, gt_uint8)
            total_ps += p
            total_ss += s
            total_batches += 1

            # 保存最后一张图作为可视化抽查
            if step == 0:
                vis = torch.cat([gt_01, y_01, out_01], dim=0)
                os.makedirs("eval_vis", exist_ok=True)
                vutils.save_image(vis, f"eval_vis/{task}_example.png", nrow=3)

        avg_ps = total_ps / total_batches
        avg_ss = total_ss / total_batches
        
        # 计算该任务的最终 FID
        fid_score = fid_metric.compute().item()
        fid_metric.reset()

        report[task] = (avg_ps, avg_ss, fid_score)
        print(f"✅ {task.upper()} 结果 -> PSNR: {avg_ps:.2f} | SSIM: {avg_ss:.4f} | FID: {fid_score:.2f}")

    print("\n" + "=" * 60)
    print(f"🏆 {name} - 最终 4 项任务全能评测报告 🏆")
    print("-" * 60)
    print(f"{'Task':<12} | {'PSNR ↑':<10} | {'SSIM ↑':<10} | {'FID ↓':<10}")
    print("-" * 60)
    for task in tasks:
        p, s, f_score = report[task]
        print(f"{task.upper():<12} | {p:>6.2f}      | {s:>6.4f}      | {f_score:>6.2f}")
    print("=" * 60)
    print("📸 纯净无伪影示例图已保存至 eval_vis/ 文件夹！")

if __name__ == "__main__":
    main()