import os
import yaml
import argparse
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import torch.utils.data as data
from tqdm import tqdm
import numpy as np

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# 导入你的核心模块
from guided_diffusion.icsa_net import ICSA_Net
from datasets import get_dataset, data_transform

# ---------------------------------------------------------
# 🌟 核心：为 Blind Deblurring 量身定制的课程式学习退化
# ---------------------------------------------------------
def generate_curriculum_blur(clean_imgs, current_epoch, total_epochs, device):
    """
    随着 epoch 增加，逐渐增大模糊的极限难度
    """
    B, C, H, W = clean_imgs.shape
    Apy = torch.zeros_like(clean_imgs)
    
    # 计算当前进度 (0.0 ~ 1.0)
    progress = min(current_epoch / total_epochs, 1.0)
    
    # 动态上限：Kernel 从 5 逐渐扩大到 13，Sigma 从 1.5 扩大到 3.5
    max_kernel = int(5 + progress * 8) 
    max_kernel = max_kernel if max_kernel % 2 != 0 else max_kernel + 1
    max_sigma = 1.5 + progress * 2.0   
    
    for i in range(B):
        # 10% 的概率保持原图，防止模型忘记清晰的特征
        if np.random.rand() < 0.1:
            Apy[i:i+1] = clean_imgs[i:i+1]
            continue
            
        # 在动态范围内随机采样当前图像的模糊难度
        kernel_size = np.random.choice(range(3, max_kernel + 1, 2))
        sigma = np.random.uniform(0.5, max_sigma)
        
        # 施加高斯模糊
        blur = TF.gaussian_blur(clean_imgs[i:i+1], kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
        
        # 注入极其微小的相机底噪 (模拟真实世界，防止过拟合到纯数学平滑)
        noise = torch.randn_like(blur) * 0.01 
        blur = blur + noise
        
        Apy[i:i+1] = torch.clamp(blur, -1.0, 1.0)
        
    return Apy

def get_alphas_cumprod(beta_start=0.0001, beta_end=0.02, num_timesteps=1000, device='cuda'):
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="celeba_hq.yml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80) # Blur 需要更长时间收敛
    
    # 【已修复】：补充了 exp 参数，解决数据集加载报错
    parser.add_argument("--exp", type=str, default="exp") 
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(os.path.join("configs", args.config), "r") as f:
        config_dict = yaml.safe_load(f)
    class Struct:
        def __init__(self, **entries): self.__dict__.update(entries)
    config = Struct(**{k: Struct(**v) if isinstance(v, dict) else v for k, v in config_dict.items()})
    config.data.random_flip = True 

    print("===== 准备数据集 =====")
    args.path_y = "celeba_hq" 
    dataset, _ = get_dataset(args, config)
    dataloader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)

    model = ICSA_Net(config).to(device)
    
    # 优化器设置
    base_lr = args.lr
    encoder_params = list(model.encoder.parameters()) + list(model.inject_conv.parameters()) + list(model.hatb_block.parameters())
    if hasattr(model, 'zero_conv'):
        encoder_params += list(model.zero_conv.parameters())
    unet_params = list(model.unet.parameters())
    
    optimizer = AdamW([
        {'params': encoder_params, 'lr': base_lr, 'name': 'encoder'},
        {'params': unet_params, 'lr': 0.0, 'name': 'unet'}
    ], weight_decay=1e-4)

    alphas_cumprod = get_alphas_cumprod(num_timesteps=config.diffusion.num_diffusion_timesteps, device=device)

    # 专属的 Blur 权重文件夹
    log_dir = "exp/logs/icsa_deblur"
    os.makedirs(log_dir, exist_ok=True)
    start_epoch = 0
    FREEZE_EPOCHS = 10 

    latest_ckpt_path = os.path.join(log_dir, "icsa_blur_latest.pt")
    if os.path.exists(latest_ckpt_path):
        print("检测到最新进度：恢复 Deblur 训练...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 
    else:
        # 如果没有 Blur 专属权重，可以先加载你之前多任务的 latest 权重做热启动
        print("首次训练 Deblur，尝试加载通用预训练权重...")
        generic_ckpt = "exp/logs/icsa_net/icsa_latest.pt"
        if os.path.exists(generic_ckpt):
            model.load_state_dict(torch.load(generic_ckpt, map_location=device)['model_state_dict'])
            print("成功加载多任务底座，将在此基础上专攻 Blur！")
        else:
            print("未找到多任务底座，将从头开始训练 U-Net！")

    model.train()
    print(f"===== 专攻 Blind Deblur 训练开始：第 {start_epoch + 1} 轮 =====")
    
    for epoch in range(start_epoch, args.epochs):
        
        # U-Net 预热解冻策略
        if epoch >= FREEZE_EPOCHS:
            for param_group in optimizer.param_groups:
                if param_group.get("name") == "unet" and param_group["lr"] == 0.0:
                    param_group["lr"] = base_lr * 0.1
                    print(f">>> U-Net 已解冻，lr = {base_lr * 0.1} <<<")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (clean_img, _) in enumerate(pbar):
            clean_img = data_transform(config, clean_img.to(device))
            B = clean_img.shape[0]

            # 应用基于当前 Epoch 难度的动态模糊
            with torch.no_grad():
                Apy = generate_curriculum_blur(clean_img, epoch, args.epochs, device)

            t = torch.randint(0, config.diffusion.num_diffusion_timesteps, (B,), device=device).long()
            noise = torch.randn_like(clean_img)
            a_t = alphas_cumprod[t].view(B, 1, 1, 1)
            xt = torch.sqrt(a_t) * clean_img + torch.sqrt(1 - a_t) * noise

            optimizer.zero_grad()
            predicted_noise = model(xt, t, Apy=Apy) 

            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # 保存权重
        latest_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        torch.save(latest_dict, latest_ckpt_path)

        if (epoch + 1) % 5 == 0:
            torch.save(latest_dict, os.path.join(log_dir, f"icsa_blur_epoch_{epoch+1}.pt"))
            print(f"✅ 已保存周期性权重: icsa_blur_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()