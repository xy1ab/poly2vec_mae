import os
import sys
from pathlib import Path
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import Subset
import time

if __package__ in {None, ""}:
    _CURRENT_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _CURRENT_DIR.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    import importlib

    V2Dataset = importlib.import_module("downstream_unet.loaders.loader").V2Dataset
else:
    from .loaders.loader import V2Dataset

def calculate_metrics(pred, target, input_blur, threshold=0.5):
    """计算硬核数据指标"""
    pred_bin = (pred > threshold).astype(np.float32)
    
    # 1. IoU
    intersection = np.sum(pred_bin * target)
    union = np.sum(pred_bin) + np.sum(target) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # 2. MSE 改善对比 (均方误差)
    mse_before = np.mean((input_blur - target) ** 2)
    mse_after = np.mean((pred - target) ** 2)
    
    # 3. 绝对错判的像素个数 
    wrong_pixels = np.sum(np.abs(pred_bin - target))
    
    # 4. 犹豫像素数 (即数值在 0.1 到 0.9 之间的模糊灰色地带)
    uncertain_before = np.sum((input_blur > 0.1) & (input_blur < 0.9))
    uncertain_after = np.sum((pred > 0.1) & (pred < 0.9))
    
    return iou, mse_before, mse_after, wrong_pixels, uncertain_before, uncertain_after

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    current_time = time.strftime("%Y%m%d_%H%M%S")
    
    # 1. 路径配置
    data_dir = './data/v2_dataset_full'
    indices_path = './data/v2_test_indices.pt'
    model_path = './checkpoints_v2/v2_unet_best.pth'

    vis_dir = './vis_results_v2'
    os.makedirs(vis_dir, exist_ok=True)
    report_path = os.path.join(vis_dir, f'evaluation_report_{current_time}.txt') 
    
    # 2. 严谨加载测试集
    full_dataset = V2Dataset(data_dir)
    test_indices = torch.load(indices_path)
    test_set = Subset(full_dataset, test_indices)
    print(f"\n📦 成功解封测试集！共抽取 {len(test_set)} 道陌生样本。")

    # 3. 加载模型
    model = smp.Unet(
        encoder_name="resnet34", encoder_weights=None,
        in_channels=3, classes=1, activation='sigmoid'
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("🤖 模型加载完毕，开始进行数据分析...\n" + "="*50)

    # 4. 随机抽取测试样本
    num_samples = 128
    sample_indices = random.sample(range(len(test_set)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(22, 6 * num_samples))
    plt.subplots_adjust(hspace=0.4)
    
    total_iou = 0.0 # 用于计算平均分
    
    # 📝 打开文件准备写入报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("      Poly2Vec 频域引导模型 - 修复评估报告      \n")
        f.write("==================================================\n\n")
        
        for i, idx in enumerate(sample_indices):
            x, y = test_set[idx]
            
            # 推理
            t_s = time.time()
            with torch.no_grad():
                pred = model(x.unsqueeze(0).to(device))
            t_d = time.time()
            print("128 time", t_d - t_s)   
            # 转换为 numpy
            img_blur = x[0].numpy()  # 通道 1: 空间模糊图
            img_mag = x[1].numpy()   # 通道 2: 频率幅值
            img_pred = pred.squeeze().cpu().numpy()
            img_gt = y.squeeze().numpy()
            
            iou, mse_bef, mse_aft, wrong_px, unc_bef, unc_aft = calculate_metrics(img_pred, img_gt, img_blur)
            total_iou += iou
            
            # 构造单条报告文本
            report_str = f"📊[测试样本 Index: {idx}]\n"
            report_str += f"   ▶ IoU (重合度)      : {iou * 100:.2f}%\n"
            report_str += f"   ▶ 像素误差 (MSE)    : 从 {mse_bef:.4f} 降至 {mse_aft:.4f} (下降了 {mse_bef/(mse_aft+1e-8):.1f} 倍!)\n"
            report_str += f"   ▶ 错判像素数        : 仅 {int(wrong_px)} 个点画错 (总像素 65536)\n"
            report_str += f"   ▶ 边缘模糊点(灰色)  : 从 {int(unc_bef)} 个锐化至 {int(unc_aft)} 个 (直角恢复)\n"
            report_str += "-" * 50 + "\n"
            
            # 同时输出到终端和文件
            print(report_str, end="")
            f.write(report_str)
            
            # --- 绘图 ---
            axes[i, 0].imshow(img_blur, cmap='gray')
            axes[i, 0].set_title(f"Input Blur\nMSE: {mse_bef:.4f} | Unclear Px: {int(unc_bef)}")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(img_mag, cmap='viridis')
            axes[i, 1].set_title("Frequency Magnitude\n(Model's X-Ray Vision)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(img_pred, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f"Prediction\nMSE: {mse_aft:.4f} | Unclear Px: {int(unc_aft)}\nIoU: {iou*100:.2f}%")
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title(f"Ground Truth\nWrong Pixels: {int(wrong_px)}/65536")
            axes[i, 3].axis('off')

        # 写入总结
        avg_iou = total_iou / num_samples
        summary_str = f"\n🎯 总结: 本次抽测 {num_samples} 个样本的平均 IoU 达到了 {avg_iou * 100:.2f}%！\n"
        print(summary_str)
        f.write(summary_str)

    img_save_path = os.path.join(vis_dir, f'test_hardcore_report_{current_time}.png')
    plt.savefig(img_save_path, bbox_inches='tight', dpi=150)
    print(f"✅ 可视化数据图片已生成: {img_save_path}")
    print(f"📄 文本评估报告已保存: {report_path}")

if __name__ == '__main__':
    main()
