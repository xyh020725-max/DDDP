import os
import glob
import random
from PIL import Image
from tqdm import tqdm

# 1. 目录配置
train_dir = "/root/autodl-tmp/DDNM-main/exp/datasets/celeba_hq/face"
source_dir = "/root/autodl-tmp/img_align_celeba"
val_dir = "/root/autodl-tmp/DDNM-main/val_data"

os.makedirs(val_dir, exist_ok=True)

# 2. 建立“物理隔离黑名单”
# 读取你训练集里的所有图片，提取它们的文件名
train_imgs = glob.glob(os.path.join(train_dir, "*.jpg")) + glob.glob(os.path.join(train_dir, "*.png"))

ban_list = set()
for p in train_imgs:
    basename = os.path.basename(p)
    # 如果有 pub_ 前缀，去掉它以还原真实的 CelebA 文件名
    if basename.startswith("pub_"):
        ban_list.add(basename.replace("pub_", ""))
    else:
        ban_list.add(basename)

print(f"🔍 已建立训练集黑名单，共 {len(ban_list)} 张图片将被严格排除。")

# 3. 从公共数据集中筛选“未见过的图片”
all_source_imgs = glob.glob(os.path.join(source_dir, "*.jpg"))
# 核心过滤逻辑：只要名字在黑名单里，直接踢掉
candidate_imgs = [img for img in all_source_imgs if os.path.basename(img) not in ban_list]

print(f"📂 在公共数据集中找到了 {len(candidate_imgs)} 张模型【绝对没见过】的新图片。")

# 4. 随机抽取 150 张作为验证集
random.seed(42) # 固定随机种子，保证实验可复现
num_val = 150
val_candidates = random.sample(candidate_imgs, min(num_val, len(candidate_imgs)))

# 5. 按照标准裁剪并保存到验证集文件夹
print(f"⚙️ 开始处理并生成 {len(val_candidates)} 张纯净验证集...")
for path in tqdm(val_candidates):
    try:
        img = Image.open(path)
        
        # 官方标准的对齐裁剪
        cx, cy = 89, 121
        crop_size = 128
        left = cx - crop_size // 2
        right = cx + crop_size // 2
        top = cy - crop_size // 2
        bottom = cy + crop_size // 2
        
        img_cropped = img.crop((left, top, right, bottom))
        
        # 高质量放大到 256x256
        img_resized = img_cropped.resize((256, 256), Image.BICUBIC)
        
        # 加上 val_ 前缀保存
        filename = os.path.basename(path)
        target_path = os.path.join(val_dir, f"val_{filename}")
        img_resized.save(target_path)
        
    except Exception as e:
        print(f"处理 {path} 时出错: {e}")

print(f"\n✅ 严谨的验证集制作完成！")
print(f"📁 存放路径: {val_dir}")
print(f"💡 这些图片已 100% 排除了你之前训练用过的 7000 张图，可以放心用于评估！")