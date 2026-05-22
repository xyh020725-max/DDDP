import os
import glob
import yaml
import argparse
import math
import torch
import torch.fft
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as calculate_ssim

# 导入你的模型
from guided_diffusion.icsa_net import ICSA_Net

# ---------------------------------------------------------
# 1. 验证集 Dataset (加入 [-1, 1] 归一化)
# ---------------------------------------------------------
class EvalDataset(Dataset):
    def __init__(self, img_dir, max_imgs=20): 
        super().__init__()
        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])[:max_imgs]
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 转到 [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 【修复核心】转到 [-1, 1]
        ])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

# 将 [-1, 1] 转换回 [0, 1] 以供保存和计算 PSNR
def unnormalize(tensor):
    return torch.clamp((tensor + 1.0) / 2.0, 0.0, 1.0)

# ---------------------------------------------------------
# 2. 确定性退化函数 (在 [-1, 1] 空间下操作)
# ---------------------------------------------------------
def generate_deterministic_degradation(clean_imgs, task_type, device):
    B, C, H, W = clean_imgs.shape
    Apy = torch.zeros_like(clean_imgs)
    
    if task_type == 'sr':
        scale = 4
        A_pool = torch.nn.AdaptiveAvgPool2d((H//scale, W//scale))
        down_img = A_pool(clean_imgs)
        Apy = F.interpolate(down_img, size=(H, W), mode='nearest')
        
    elif task_type == 'cs':
        keep_ratio = 0.3
        freq_img = torch.fft.fft2(clean_imgs, norm="ortho")
        mask = torch.zeros((1, 1, H, W), device=device)
        cy, cx = H // 2, W // 2
        keep_h, keep_w = int(H * math.sqrt(keep_ratio)), int(W * math.sqrt(keep_ratio))
        mask[:, :, cy-keep_h//2 : cy+keep_h//2, cx-keep_w//2 : cx+keep_w//2] = 1.0 
        
        freq_shifted = torch.fft.fftshift(freq_img)
        freq_img_masked = torch.fft.ifftshift(freq_shifted * mask)
        Apy = torch.fft.ifft2(freq_img_masked, norm="ortho").real
        
    elif task_type == 'inpainting':
        # 在 [-1, 1] 空间中，通常用 0 (灰色) 填充遮挡区域
        mask = torch.ones((1, 1, H, W), device=device)
        cy, cx = H // 2, W // 2
        mask[:, :, cy-32:cy+32, cx-32:cx+32] = 0
        Apy = clean_imgs * mask
        
    elif task_type == 'blur':
        Apy = TF.gaussian_blur(clean_imgs, kernel_size=[9, 9], sigma=[2.0, 2.0])
        
    return Apy

# ---------------------------------------------------------
# 3. DDIM 快速采样器 (修正 clamp 范围为 [-1, 1])
# ---------------------------------------------------------
def get_alphas_cumprod(beta_start=0.0001, beta_end=0.02, num_timesteps=1000, device='cuda'):
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)

@torch.inference_mode() # 使用更快的 inference_mode
def ddim_sample(model, Apy, config, device, steps=50):
    B, C, H, W = Apy.shape
    xt = torch.randn((B, C, H, W), device=device)
    
    alphas_cumprod = get_alphas_cumprod(device=device)
    total_steps = config.diffusion.num_diffusion_timesteps
    times = torch.linspace(total_steps - 1, 0, steps, device=device).long()
    
    for i in range(steps):
        t = times[i]
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        
        noise_pred = model(xt, t_batch, Apy=Apy)
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[times[i+1]] if i < steps - 1 else torch.tensor(1.0, device=device)
        
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        # 【修复核心】模型预测出的 x0 应该在 [-1, 1] 之间
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        
        if i < steps - 1:
            xt = torch.sqrt(alpha_t_prev) * x0_pred + torch.sqrt(1 - alpha_t_prev) * noise_pred
            
    return x0_pred

# ---------------------------------------------------------
# 4. 评价指标计算
# ---------------------------------------------------------
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()

def compute_metrics(pred_tensor, gt_tensor):
    # 此时传入的应该是已经转换到 [0, 1] 的 tensor
    pred_np = pred_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    gt_np = gt_tensor.cpu().numpy().transpose(0, 2, 3, 1)
    
    batch_psnr, batch_ssim = 0.0, 0.0
    B = pred_np.shape[0]
    
    for i in range(B):
        batch_psnr += calculate_psnr(pred_tensor[i], gt_tensor[i])
        batch_ssim += calculate_ssim(gt_np[i], pred_np[i], data_range=1.0, channel_axis=-1)
        
    return batch_psnr / B, batch_ssim / B

# ---------------------------------------------------------
# 5. 主程序
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="celeba_hq.yml")
    parser.add_argument("--val_dir", type=str, default="val_data")
    parser.add_argument("--log_dir", type=str, default="exp/logs/icsa_net")
    parser.add_argument("--num_imgs", type=int, default=8, help="先只测 8 张图快速排错")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join("configs", args.config), "r") as f:
        config_dict = yaml.safe_load(f)
    class Struct:
        def __init__(self, **entries): self.__dict__.update(entries)
    config = Struct(**{k: Struct(**v) if isinstance(v, dict) else v for k, v in config_dict.items()})

    print(f"===== 准备评测 ({args.val_dir}) =====")
    val_dataset = EvalDataset(args.val_dir, max_imgs=args.num_imgs)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    model = ICSA_Net(config).to(device)
    
    pt_files = glob.glob(os.path.join(args.log_dir, "*.pt"))
    pt_files.sort()
    
    # 强烈建议：目前只测最新的两个权重，快速定位问题
    pt_files = pt_files[-2:] 
    
    tasks = ['sr', 'cs', 'inpainting', 'blur']
    final_report = {}
    
    # 创建可视化大图保存目录
    vis_dir = "eval_visuals"
    os.makedirs(vis_dir, exist_ok=True)

    for pt_file in pt_files:
        ckpt_name = os.path.basename(pt_file)
        print(f"\n🚀 正在评估权重: {ckpt_name}")
        
        checkpoint = torch.load(pt_file, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        
        final_report[ckpt_name] = {}
        
        for task in tasks:
            total_psnr, total_ssim = 0.0, 0.0
            print(f"  👉 测试任务: {task.upper()}")
            
            # 创建该任务专属的图片保存目录
            task_vis_dir = os.path.join(vis_dir, ckpt_name.replace('.pt', ''), task)
            os.makedirs(task_vis_dir, exist_ok=True)
            
            for step, clean_imgs in enumerate(tqdm(val_loader, desc=task, leave=False)):
                clean_imgs = clean_imgs.to(device)
                
                Apy = generate_deterministic_degradation(clean_imgs, task, device)
                restored_imgs = ddim_sample(model, Apy, config, device, steps=50)
                
                # 【关键】将 [-1, 1] 转换回 [0, 1]
                restored_01 = unnormalize(restored_imgs)
                clean_01 = unnormalize(clean_imgs)
                Apy_01 = unnormalize(Apy)
                
                # 计算指标
                psnr, ssim = compute_metrics(restored_01, clean_01)
                total_psnr += psnr
                total_ssim += ssim
                
                # 【救命神技】把原图、退化图、修复图拼成一行保存下来
                # 第一行: 原图 | 第二行: 损坏图 | 第三行: 修复图
                comparison = torch.cat([clean_01, Apy_01, restored_01], dim=0)
                vutils.save_image(comparison, os.path.join(task_vis_dir, f"batch_{step}.png"), nrow=clean_imgs.shape[0])
                
            avg_psnr = total_psnr / len(val_loader)
            avg_ssim = total_ssim / len(val_loader)
            final_report[ckpt_name][task] = {"PSNR": avg_psnr, "SSIM": avg_ssim}
            
            print(f"     结果 -> PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.4f}")

    print("\n" + "="*60)
    print("🏆 修正后的指标对比报告 🏆")
    print("-" * 60)
    for ckpt, results in final_report.items():
        for task in tasks:
            p = results[task]['PSNR']
            s = results[task]['SSIM']
            print(f"{ckpt:<20} | {task.upper():<10} | {p:>6.2f}   | {s:>6.4f}")
    print("="*60)
    print(f"\n📸 【重要】请立即前往 {vis_dir} 文件夹查看保存的对比图！")

if __name__ == "__main__":
    main())

if __name__ == "__main__":
    main()