import os
import glob
import random
import shutil
from tqdm import tqdm

def main():
    # 1. 原始混合数据所在目录 (请根据你的实际情况核对)
    source_dir = "/root/autodl-tmp/DDNM-main/exp/datasets/celeba_hq/face"
    
    # 2. 我们将创建一个全新的、规范的数据集根目录
    base_target_dir = "/root/autodl-tmp/DDNM-main/exp/datasets/celeba_hq_standard"
    
    train_dir = os.path.join(base_target_dir, "train", "face")
    val_dir = os.path.join(base_target_dir, "val", "face")
    test_dir = os.path.join(base_target_dir, "test", "face")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 3. 读取所有图片
    all_imgs = glob.glob(os.path.join(source_dir, "*.jpg")) + glob.glob(os.path.join(source_dir, "*.png"))
    
    # 分离扩充图(pub_)和原版高清图
    pub_imgs = [img for img in all_imgs if os.path.basename(img).startswith("pub_")]
    hq_imgs = [img for img in all_imgs if not os.path.basename(img).startswith("pub_")]
    
    print(f"找到扩充图像 (pub_): {len(pub_imgs)} 张")
    print(f"找到高清原图 (HQ): {len(hq_imgs)} 张")

    # 4. 随机打乱高清原图
    random.seed(42) # 固定随机种子，保证每次划分一致
    random.shuffle(hq_imgs)

    # 5. 划分数量：500测试，500验证，剩下全给训练
    num_test = 500
    num_val = 500
    
    test_list = hq_imgs[:num_test]
    val_list = hq_imgs[num_test : num_test + num_val]
    train_hq_list = hq_imgs[num_test + num_val :]
    
    # 训练集 = 剩下的 HQ 原图 + 所有的 pub_ 扩充图
    train_list = train_hq_list + pub_imgs

    # 6. 开始物理转移 (使用 copy2 复制，保留原文件夹作为备份)
    print("\n🚀 开始构建 [测试集 Test]...")
    for img in tqdm(test_list):
        shutil.copy2(img, os.path.join(test_dir, os.path.basename(img)))

    print("\n🚀 开始构建 [验证集 Val]...")
    for img in tqdm(val_list):
        shutil.copy2(img, os.path.join(val_dir, os.path.basename(img)))

    print("\n🚀 开始构建 [训练集 Train]...")
    for img in tqdm(train_list):
        shutil.copy2(img, os.path.join(train_dir, os.path.basename(img)))

    print(f"\n✅ 数据集规范化完成！")
    print(f"📁 训练集: {len(train_list)} 张 -> {train_dir}")
    print(f"📁 验证集: {len(val_list)} 张 -> {val_dir}")
    print(f"📁 测试集: {len(test_list)} 张 -> {test_dir}")

if __name__ == "__main__":
    main()