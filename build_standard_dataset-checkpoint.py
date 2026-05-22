import os
import glob
import shutil
from PIL import Image
from tqdm import tqdm

def process_and_save(img_paths, target_dir, prefix=""):
    os.makedirs(target_dir, exist_ok=True)
    for path in tqdm(img_paths, desc=f"生成 {prefix} 数据"):
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
            
            filename = os.path.basename(path)
            target_path = os.path.join(target_dir, f"{prefix}{filename}")
            img_resized.save(target_path)
        except Exception as e:
            print(f"处理 {path} 时出错: {e}")

def main():
    # 1. 目录配置
    public_celeba_dir = "/root/autodl-tmp/img_align_celeba"
    old_train_dir = "/root/autodl-tmp/DDNM-main/exp/datasets/celeba_hq/face"
    
    base_target_dir = "/root/autodl-tmp/DDNM-main/exp/datasets/celeba_hq_standard"
    train_dir = os.path.join(base_target_dir, "train", "face")
    val_dir = os.path.join(base_target_dir, "val", "face")
    test_dir = os.path.join(base_target_dir, "test", "face")
    
    os.makedirs(train_dir, exist_ok=True)

    # 2. 转移现有的 30000 张训练图
    print("🚀 正在转移已有的 30000 张训练集图片...")
    existing_train_imgs = glob.glob(os.path.join(old_train_dir, "pub_*.jpg"))
    for img in tqdm(existing_train_imgs, desc="转移 Train"):
        shutil.copy2(img, os.path.join(train_dir, os.path.basename(img)))
    
    # 顺便把那 8 张官方 Demo 塞进测试集玩玩
    demo_imgs = glob.glob(os.path.join(old_train_dir, "*.png"))
    os.makedirs(test_dir, exist_ok=True)
    for img in demo_imgs:
        shutil.copy2(img, os.path.join(test_dir, os.path.basename(img)))

    # 3. 从公共数据集中获取剩余图片来制作 Val 和 Test
    all_public_imgs = glob.glob(os.path.join(public_celeba_dir, "*.jpg"))
    all_public_imgs.sort() # 确保排序一致
    
    if len(all_public_imgs) < 31000:
        print("错误：公共数据集图片不足 31000 张，请检查解压路径！")
        return

    # 因为之前你截取了前 30000 张做训练，现在我们从第 30000 张往后取
    val_source = all_public_imgs[30000:30500]
    test_source = all_public_imgs[30500:31000]

    # 4. 生成验证集和测试集
    print("\n🚀 开始制作 [验证集 Val] (500张)...")
    process_and_save(val_source, val_dir, prefix="val_")
    
    print("\n🚀 开始制作 [测试集 Test] (500张)...")
    process_and_save(test_source, test_dir, prefix="test_")

    print("\n🎉 全部数据集构建完成！")
    print(f"📁 训练集: {len(glob.glob(os.path.join(train_dir, '*')))} 张")
    print(f"📁 验证集: 500 张")
    print(f"📁 测试集: 508 张 (含 8 张官方 Demo)")

if __name__ == "__main__":
    main()