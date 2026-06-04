"""
YOLOv8 手持物品检测模型训练脚本
================================
训练用于识别驾驶场景中手持物品（手机、香烟、水杯等）的YOLOv8目标检测模型。

数据集需要按照以下结构准备:
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

运行方式:
    python -m yolo.train_handheld
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def train_model():
    """
    训练YOLOv8手持物品检测模型

    训练参数说明:
    - 模型: yolov8n.pt (nano版本，最轻量，适合CPU训练)
    - epochs: 50 (训练轮数)
    - imgsz: 320 (输入图像尺寸，降低以加速训练)
    - batch: 8 (批次大小，CPU训练使用较小值)
    - workers: 0 (Windows兼容模式)
    - device: 'cpu' (CPU训练)
    - patience: 10 (早停，10轮无提升则停止)
    """
    from ultralytics import YOLO

    # 数据集配置文件
    data_yaml = BASE_DIR / 'dataset' / 'data.yaml'

    if not data_yaml.exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        print("请先运行 dataset_prep.py 准备数据集")
        print(f"或手动创建 {data_yaml}")
        return None

    print("=" * 60)
    print("YOLOv8 手持物品检测模型训练")
    print("=" * 60)
    print(f"数据集配置: {data_yaml}")
    print("模型: yolov8n.pt (预训练权重)")
    print("设备: CPU")
    print("训练轮数: 50")
    print("图像尺寸: 320")
    print("批次大小: 8")
    print("=" * 60)

    # 加载预训练模型
    model = YOLO('yolov8n.pt')

    # 开始训练
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=320,
            batch=8,
            workers=0,
            device='cpu',
            patience=10,
            name='handheld_detection',
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            augment=True,
        )

        print("\n[训练完成]")
        print(f"最佳模型保存在: {results.save_dir}")

        # 导出ONNX模型（可选，用于部署）
        best_model_path = Path(results.save_dir) / 'weights' / 'best.pt'
        if best_model_path.exists():
            print("正在导出ONNX模型...")
            best_model = YOLO(str(best_model_path))
            best_model.export(format='onnx')

            # 复制最佳模型到models_data目录
            import shutil
            dest_path = BASE_DIR / 'models_data' / 'yolo_handheld.pt'
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(best_model_path, dest_path)
            print(f"模型已复制到: {dest_path}")

        return results

    except Exception as e:
        print(f"\n[训练出错] {e}")
        import traceback
        traceback.print_exc()
        return None


def quick_train():
    """
    快速训练（用于测试和验证数据集配置是否正确）

    使用更少的轮数和更小的图像尺寸进行快速验证
    """
    from ultralytics import YOLO

    data_yaml = BASE_DIR / 'dataset' / 'data.yaml'

    if not data_yaml.exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        return None

    print("快速验证训练 (5 epochs, imgsz=224, batch=4)...")

    model = YOLO('yolov8n.pt')

    results = model.train(
        data=str(data_yaml),
        epochs=5,
        imgsz=224,
        batch=4,
        workers=0,
        device='cpu',
        name='handheld_quick_test',
        exist_ok=True,
    )

    print(f"快速训练完成，模型保存在: {results.save_dir}")
    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='训练YOLOv8手持物品检测模型')
    parser.add_argument('--quick', action='store_true', help='快速验证训练 (5轮)')
    args = parser.parse_args()

    if args.quick:
        quick_train()
    else:
        train_model()
