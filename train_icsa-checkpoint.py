import os
import yaml
import argparse
import torch
import torch.fft
import torch.nn.functional as F
from torch.optim import AdamW
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 导入你修改过的模块
from guided_diffusion.icsa_net import ICSA_Net
from datasets import get_dataset, data_transform
from functions.ckpt_util import download
from functions.svd_operators import SuperResolution, WalshHadamardCS, Inpainting, Deblurring

# ---------------------------------------------------------
# 1. 官方 DDNM SVD 退化管理器 (全局缓存，防止训练卡死)
# ---------------------------------------------------------
class OfficialDegradationManager:
    def __init__(self, config, device):
        self.device = device
        self.C = config.data.channels
        self.H = config.data.image_size
        
        print("⏳ 正在初始化 DDNM 官方 SVD 退化算子 (这可能需要几十秒钟，仅执行一次)...")
        
        # 1. 超分 (4x SR)
        self.sr_op = SuperResolution(self.C, self.H, 4, device)
        
        # 2. 压缩感知 (WH-CS 25%)
        # 固定随机种子，保证训练和测试的 Measurement Matrix 完美一致
        torch.manual_seed(42)
        self.cs_perm = torch.randperm(self.H**2, device=device)
        self.cs_op = WalshHadamardCS(self.C, self.H, 4, self.cs_perm, device)
        
        # 3. 图像修复 (Inpainting)
        mask_path = "exp/inp_masks/mask.npy"
        if os.path.exists(mask_path):
            loaded = np.load(mask_path)
            mask = torch.from_numpy(loaded).to(device).reshape(-1)
            missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            self.inp_missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
        else:
            mask = torch.ones(self.C, self.H, self.H)
            mask[:, self.H//2-32:self.H//2+32, self.H//2-32:self.H//2+32] = 0
            self.inp_missing = torch.nonzero(mask.flatten() == 0).squeeze().long().to(device)
            
        self.inp_op = Inpainting(self.C, self.H, self.inp_missing, device)
        
        # 4. 去模糊 (5x5 Gauss Blur, sigma=10)
        sigma = 10
        pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x / sigma) ** 2]))
        kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(device)
        self.blur_op = Deblurring(kernel / kernel.sum(), self.C, self.H, device)
        
        print("✅ 官方 SVD 算子初始化完成！")

    def apply(self, x, task):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, -1)
        
        if task == 'sr':
            y = self.sr_op.A(x_flat)
            Apy = self.sr_op.A_pinv(y).view(B, C, H, W)
            
        elif task == 'cs':
            y = self.cs_op.A(x_flat)
            Apy = self.cs_op.A_pinv(y).view(B, C, H, W)
            
        elif task == 'inpainting':
            y = self.inp_op.A(x_flat)
            Apy = self.inp_op.A_pinv(y).view(B, C, H, W)
            ones = torch.ones_like(x).reshape(B, -1)
            Apy += self.inp_op.A_pinv(self.inp_op.A(ones)).view(B, C, H, W) - 1
            
        elif task == 'blur':
            y = self.blur_op.A(x_flat)
            Apy = y.view(B, C, H, W) 
            
        elif task == 'clean':
            y = x
            Apy = x
            
        else:
            raise ValueError(f"Unknown task: {task}")
            
        return y, Apy

# ---------------------------------------------------------
# 2. 获取加噪时间步的系数
# ---------------------------------------------------------
def get_alphas_cumprod(beta_start=0.0001, beta_end=0.02, num_timesteps=1000, device='cuda'):
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)

# ---------------------------------------------------------
# 3. 主训练循环
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="celeba_hq.yml")
    parser.add_argument("--exp", type=str, default="exp")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 解析配置文件
    with open(os.path.join("configs", args.config), "r") as f:
        config_dict = yaml.safe_load(f)
    class Struct:
        def __init__(self, **entries): self.__dict__.update(entries)
    config = Struct(**{k: Struct(**v) if isinstance(v, dict) else v for k, v in config_dict.items()})
    config.data.random_flip = True 

    # 1. 准备数据
    print("===== 准备数据集 =====")
    args.path_y = "celeba_hq" 
    dataset, _ = get_dataset(args, config)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    # ========================================================
    # 【新增】：全局实例化退化管理器 (极其重要，不可放入循环)
    # ========================================================
    deg_manager = OfficialDegradationManager(config, device)

    # 2. 初始化模型
    model = ICSA_Net(config).to(device)
    
    # 3. 分组优化器：使用精准的 `name` 标签区分参数组
    base_lr = args.lr
    encoder_params = list(model.encoder.parameters()) + list(model.inject_conv.parameters()) + list(model.hatb_block.parameters())
    
    # 将零卷积也加入优化组（如果你按照我的建议命名为 zero_conv 的话）
    if hasattr(model, 'zero_conv'):
        encoder_params += list(model.zero_conv.parameters())
        
    unet_params = list(model.unet.parameters())
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': base_lr, 'name': 'encoder'},
        {'params': unet_params, 'lr': 0.0, 'name': 'unet'}
    ], weight_decay=1e-4)

    alphas_cumprod = get_alphas_cumprod(num_timesteps=config.diffusion.num_diffusion_timesteps, device=device)

    # 4. 断点与预训练权重加载逻辑
    log_dir = "exp/logs/icsa_net"
    os.makedirs(log_dir, exist_ok=True)
    start_epoch = 0
    FREEZE_EPOCHS = 10 # 前10个Epoch冻结U-Net

    latest_ckpt_path = os.path.join(log_dir, "icsa_latest.pt")
    existing_ckpts = [f for f in os.listdir(log_dir) if f.endswith('.pt') and 'icsa_epoch' in f]

    if os.path.exists(latest_ckpt_path):
        print("检测到最新进度：正在从 icsa_latest.pt 恢复训练...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 
        else:
            model.load_state_dict(checkpoint)

    elif len(existing_ckpts) > 0:
        epochs_saved = [int(f.split('_')[-1].split('.')[0]) for f in existing_ckpts]
        latest_epoch = max(epochs_saved)
        ckpt_path = os.path.join(log_dir, f"icsa_epoch_{latest_epoch}.pt")
        
        print(f"未找到 latest，降级处理：正在从 Epoch {latest_epoch} 恢复训练...")
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1 
        else:
            model.load_state_dict(checkpoint)
            start_epoch = latest_epoch 
            
    else:
        print("未发现本地进度，加载官方权重执行热启动...")
        official_path = os.path.join(args.exp, "logs/celeba/celeba_hq.ckpt")
        if not os.path.exists(official_path):
            download('https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt', official_path)
        
        pretrained_state_dict = torch.load(official_path, map_location=device)
        missing, unexpected = model.unet.load_state_dict(pretrained_state_dict, strict=False)
        print(f"U-Net 预训练权重已加载。未匹配的新增层数量: {len(missing)}")

    # 5. 开始训练
    model.train()
    print(f"===== 训练开始：从第 {start_epoch + 1} 轮开始 =====")
    
    for epoch in range(start_epoch, args.epochs):
        
        # --- 安全解冻策略：通过 name 精确制导 ---
        if epoch >= FREEZE_EPOCHS:
            for param_group in optimizer.param_groups:
                if param_group.get("name") == "unet" and param_group["lr"] == 0.0:
                    param_group["lr"] = base_lr * 0.1
                    print(f">>> U-Net 已解冻，lr = {base_lr * 0.1} <<<")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (clean_img, _) in enumerate(pbar):
            clean_img = data_transform(config, clean_img.to(device))
            B = clean_img.shape[0]

            # ========================================================
            # 【核心修改】：通过退化管理器生成网络输入 Apy
            # ========================================================
            task_type = np.random.choice(['sr', 'cs', 'inpainting', 'blur', 'clean'], 
                                         p=[0.25, 0.25, 0.25, 0.2, 0.05])
            with torch.no_grad():
                _, Apy = deg_manager.apply(clean_img, task_type)

            t = torch.randint(0, config.diffusion.num_diffusion_timesteps, (B,), device=device).long()

            # 加噪
            noise = torch.randn_like(clean_img)
            a_t = alphas_cumprod[t].view(B, 1, 1, 1)
            xt = torch.sqrt(a_t) * clean_img + torch.sqrt(1 - a_t) * noise

            # 预测噪声
            optimizer.zero_grad()
            predicted_noise = model(xt, t, Apy=Apy) 

            # 反向传播
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"Task": task_type.upper(), "Loss": f"{loss.item():.4f}"})
            
        # ---------------------------------------------------------
        # 6. 保存权重
        # ---------------------------------------------------------

        # 1. 每轮覆盖保存最新权重
        latest_path = os.path.join(log_dir, "icsa_latest.pt")
        latest_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        torch.save(latest_dict, latest_path)

        # 2. 每 5 个 Epoch 持久化保存
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(log_dir, f"icsa_epoch_{epoch+1}.pt")
            torch.save(latest_dict, save_path)
            print(f"模型已保存至: {save_path}")

if __name__ == "__main__":
    main()