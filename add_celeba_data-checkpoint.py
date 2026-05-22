import os
import glob
from PIL import Image
from tqdm import tqdm

# 1. 刚刚解压出来的公共 CelebA 图片路径
source_dir = "/root/autodl-tmp/img_align_celeba" 
# 2. 你的目标训练集文件夹 (注意要放到 face 子目录下，以匹配 ImageFolder 的读取逻辑)
target_dir = "/root/autodl-tmp/DDNM-main/exp/datasets/celeba_hq/face"

os.makedirs(target_dir, exist_ok=True)

img_paths = glob.glob(os.path.join(source_dir, "*.jpg"))
img_paths.sort()

if len(img_paths) == 0:
    print("未找到图片，请检查解压的 source_dir 路径是否正确！")
    exit()

print(f"找到 {len(img_paths)} 张公共 CelebA 图片。")

# 设定你想补充的图片数量 (比如补充 30000 张来训练，可以根据你的需求调整)
num_add = 30000 
img_paths = img_paths[:num_add]

print(f"开始裁剪并转移 {len(img_paths)} 张图片到你的数据集中...")

for path in tqdm(img_paths):
    try:
        img = Image.open(path)
        
        # 按照 DDNM 官方在 datasets/__init__.py 中的标准进行面部中心裁剪
        cx, cy = 89, 121
        crop_size = 128
        left = cx - crop_size // 2
        right = cx + crop_size // 2
        top = cy - crop_size // 2
        bottom = cy + crop_size // 2
        
        img_cropped = img.crop((left, top, right, bottom))
        
        # 将裁剪后的正方形图片高质量放大到 256x256，与 CelebA-HQ 的尺寸无缝衔接
        img_resized = img_cropped.resize((256, 256), Image.BICUBIC)
        
        # 保存到你的 CelebA_HQ 文件夹，加上 'pub_' 前缀以便与原有的 HQ 图片区分
        filename = os.path.basename(path)
        target_path = os.path.join(target_dir, f"pub_{filename}")
        img_resized.save(target_path)
        
    except Exception as e:
        print(f"处理 {path} 时出错: {e}")

print("数据集扩充完成！现在你可以继续训练了。")