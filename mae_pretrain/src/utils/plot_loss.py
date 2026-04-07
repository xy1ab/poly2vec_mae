import matplotlib.pyplot as plt
import re
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="从训练日志中解析并绘制 Loss 曲线")
    parser.add_argument("--log_path", type=str, required=True, help="日志文件的路径")
    parser.add_argument("--save_path", type=str, default=None, help="图片保存路径（默认与日志同目录）")
    return parser.parse_args()

def extract_metrics(file_path):
    # 将 Epoch 轴分开存储
    data = {
        "train_epochs": [],
        "train_total": [], "train_mag": [], "train_phase": [],
        "val_epochs": [],
        "val_total": [], "val_mag": [], "val_phase": []
    }

    epoch_re = re.compile(r"--- Epoch \[(\d+)/\d+\] Started ---")
    train_re = re.compile(r"\[Train\] Total: ([\d.]+) \| Mag: ([\d.]+) \| Phase: ([\d.]+)")
    val_re = re.compile(r"\[Val\]\s+Total: ([\d.]+) \| Mag: ([\d.]+) \| Phase: ([\d.]+)")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    current_epoch = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e_match = epoch_re.search(line)
            if e_match:
                current_epoch = int(e_match.group(1))
            
            # 解析训练数据
            t_match = train_re.search(line)
            if t_match:
                data["train_epochs"].append(current_epoch)
                data["train_total"].append(float(t_match.group(1)))
                data["train_mag"].append(float(t_match.group(2)))
                data["train_phase"].append(float(t_match.group(3)))
            
            # 解析验证数据
            v_match = val_re.search(line)
            if v_match:
                data["val_epochs"].append(current_epoch)
                data["val_total"].append(float(v_match.group(1)))
                data["val_mag"].append(float(v_match.group(2)))
                data["val_phase"].append(float(v_match.group(3)))
    
    return data

def plot_metrics(data, log_path, custom_save_path):
    if custom_save_path is None:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        save_path = os.path.join(log_dir, "loss_curve.png")
    else:
        save_path = custom_save_path

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # 获取文件名作为标题
    log_name = os.path.basename(log_path)
    fig.suptitle(f"Training Metrics - {log_name}", fontsize=16)
    
    metrics = [
        ('Total Loss', 'train_total', 'val_total'),
        ('Mag Loss', 'train_mag', 'val_mag'),
        ('Phase Loss', 'train_phase', 'val_phase')
    ]
    
    for i, (title, t_key, v_key) in enumerate(metrics):
        # 绘制训练曲线：使用 train_epochs
        if data["train_epochs"]:
            axes[i].plot(data["train_epochs"], data[t_key], 
                         label='Train', color='#1f77b4', alpha=0.7)
        
        # 绘制验证曲线：使用 val_epochs (独立长度)
        if data["val_epochs"]:
            # 如果验证点很少，建议加上 marker='o' 方便查看数据点
            axes[i].plot(data["val_epochs"], data[v_key], 
                         label='Val', color='#ff7f0e', linestyle='--', marker='o', markersize=4)
        
        axes[i].set_title(title)
        axes[i].set_xlabel('Epoch')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to: {save_path}")
    print(f"Parsed {len(data['train_epochs'])} train points and {len(data['val_epochs'])} val points.")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    log_data = extract_metrics(args.log_path)
    if log_data and (log_data["train_epochs"] or log_data["val_epochs"]):
        plot_metrics(log_data, args.log_path, args.save_path)