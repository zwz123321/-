import os
import shutil
from sklearn.model_selection import train_test_split

# 原始数据路径和目标路径
source_dir = 'D:/python代码调试/项目实战/自定义数据/cat_dog'           # 替换为你的原始图片文件夹路径
target_dir = 'GoogLeNet_1\data'    # 新建的目标文件夹名

# 划分比例
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# 创建 train/val/test 文件夹下的子类文件夹
def create_class_dirs(base_dir, classes):
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            os.makedirs(class_dir, exist_ok=True)

# 复制图像到目标路径
def copy_images_to_split(source_dir, target_dir, train_list, val_list, test_list, cls):
    for img in train_list:
        shutil.copy(os.path.join(source_dir, cls, img), os.path.join(target_dir, 'train', cls))
    for img in val_list:
        shutil.copy(os.path.join(source_dir, cls, img), os.path.join(target_dir, 'val', cls))
    for img in test_list:
        shutil.copy(os.path.join(source_dir, cls, img), os.path.join(target_dir, 'test', cls))

# 主函数：划分并移动文件
def split_dataset(source_dir, target_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0, "总比例必须为 1"

    classes = os.listdir(source_dir)  # 获取所有类别文件夹
    create_class_dirs(target_dir, classes)

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = os.listdir(cls_path)
        total = len(images)

        # 使用 stratify 划分数据集
        train_val, test = train_test_split(images, test_size=test_ratio, random_state=42)
        train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=42)

        print(f"Class: {cls} -> Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

        # 移动图像
        copy_images_to_split(source_dir, target_dir, train, val, test, cls)

    print("数据集划分完成，已保存至:", target_dir)

# 运行划分
if __name__ == '__main__':
    split_dataset(
        source_dir,
        target_dir,
        train_ratio,
        val_ratio,
        test_ratio
    )