import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange
import torchvision.transforms.functional as TF

# 导入 DDNM 官方的 Model
from guided_diffusion.models import Model, get_timestep_embedding, nonlinearity

# =======================================================
# 1. 核心器官一：来自 PromptIR 的动态提示生成块 (PromptGenBlock)
# =======================================================
class PromptGenBlock(nn.Module):
    def __init__(self, in_channels=3, prompt_dim=512, prompt_len=5, prompt_size=16):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(in_channels, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B,1,1,1,1,1).squeeze(1)
        prompt = torch.sum(prompt, dim=1)
        prompt = self.conv3x3(prompt)
        return prompt

# =======================================================
# 2. 你的理论创新：双域算子感知编码器 (Dual-Domain Encoder)
# =======================================================
class DualDomainEncoder(nn.Module):
    def __init__(self, in_channels=3, prompt_dim=512):
        super().__init__()
        self.spatial_prompt = PromptGenBlock(in_channels=in_channels, prompt_dim=prompt_dim)
        self.freq_prompt = PromptGenBlock(in_channels=in_channels, prompt_dim=prompt_dim)
        self.fusion_conv = nn.Conv2d(prompt_dim * 2, prompt_dim, kernel_size=1)

    def forward(self, degraded_img):
        sp_prompt = self.spatial_prompt(degraded_img)
        freq_img = torch.fft.fft2(degraded_img, norm="ortho")
        mag = torch.abs(freq_img)
        fr_prompt = self.freq_prompt(mag)
        fused = torch.cat([sp_prompt, fr_prompt], dim=1)
        out_prompt = self.fusion_conv(fused)
        return out_prompt

# =======================================================
# 3. 核心器官二：混合注意力模块 (HATB)
# =======================================================
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        sigma = x.var(1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1)
    def forward(self, x):
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)

class MDTA_Attention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = rearrange(attn.softmax(dim=-1) @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    def forward(self, x):
        attn = torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)
        return torch.sigmoid(self.conv1(attn)) * x

class HATB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = MDTA_Attention(dim)
        self.norm2 = LayerNorm(dim)
        self.spatial_attn = SpatialAttention()
        self.norm3 = LayerNorm(dim)
        self.ffn = FeedForward(dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.spatial_attn(self.norm2(x))
        x = x + self.ffn(self.norm3(x))
        return x

# =======================================================
# 心脏重组：ICSA-Net 主干网络 (零卷积残差保护机制)
# =======================================================
class ICSA_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        deepest_ch = config.model.ch * config.model.ch_mult[-1] 
        
        # 1. 双域提示生成器 
        self.encoder = DualDomainEncoder(in_channels=3, prompt_dim=deepest_ch) 
        
        # 2. 直接使用官方底层的 Model
        self.unet = Model(config)
        
        # 3. 提取混合特征的模块 (正常随机初始化即可)
        self.inject_conv = nn.Conv2d(deepest_ch * 2, deepest_ch, kernel_size=1)
        self.hatb_block = HATB(dim=deepest_ch)

        # 4. ==========================================================
        # 【救命神技：零卷积】用于将混合好的特征，安全且无破坏地融入 U-Net
        # ==========================================================
        self.zero_conv = nn.Conv2d(deepest_ch, deepest_ch, kernel_size=1)
        # 将权重和偏置强制初始化为 0
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, x, t, **kwargs):
        degraded_img = kwargs.get('Apy', x)
        prompt = self.encoder(degraded_img)

        temb = get_timestep_embedding(t, self.unet.ch)
        temb = self.unet.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.unet.temb.dense[1](temb)

        hs = [self.unet.conv_in(x)]
        for i_level in range(self.unet.num_resolutions):
            for i_block in range(self.unet.num_res_blocks):
                h = self.unet.down[i_level].block[i_block](hs[-1], temb)
                if len(self.unet.down[i_level].attn) > 0:
                    h = self.unet.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.unet.num_resolutions-1:
                hs.append(self.unet.down[i_level].downsample(hs[-1]))

        h = hs[-1]
        h = self.unet.mid.block_1(h, temb)
        h = self.unet.mid.attn_1(h)
        h = self.unet.mid.block_2(h, temb)
        
        # === 核心注入点 (ControlNet 零卷积残差模式) ===
        prompt_resized = F.interpolate(prompt, size=h.shape[2:], mode='bilinear', align_corners=False)
        
        # 第一步：让你设计的 inject_conv 和 hatb 模块去处理特征混合
        mixed_feature = self.inject_conv(torch.cat([h, prompt_resized], dim=1))
        mixed_feature = self.hatb_block(mixed_feature)

        # 第二步：【残差相加】通过零卷积，将提取好的特征安全地累加到 h 上！
        # 在初始状态下 zero_conv 输出全是 0，h = h + 0，完美保护了预训练权重！
        h = h + self.zero_conv(mixed_feature)

        for i_level in reversed(range(self.unet.num_resolutions)):
            for i_block in range(self.unet.num_res_blocks+1):
                h = self.unet.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.unet.up[i_level].attn) > 0:
                    h = self.unet.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.unet.up[i_level].upsample(h)

        h = self.unet.norm_out(h)
        h = nonlinearity(h)
        h = self.unet.conv_out(h)
        return h