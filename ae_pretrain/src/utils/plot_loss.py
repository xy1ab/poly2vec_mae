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
    data = {
        "epoch": [],
        "train_total": [], "train_mag": [], "train_phase": [],
        "val_total": [], "val_mag": [], "val_phase": []
    }

    # 适配你日志格式的正则
    epoch_re = re.compile(r"--- Epoch \[(\d+)/\d+\] Started ---")
    train_re = re.compile(r"\[Train\] Total: ([\d.]+) \| Mag: ([\d.]+) \| Phase: ([\d.]+)")
    val_re = re.compile(r"\[Val\]\s+Total: ([\d.]+) \| Mag: ([\d.]+) \| Phase: ([\d.]+)")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            e_match = epoch_re.search(line)
            if e_match:
                current_epoch = int(e_match.group(1))
            
            t_match = train_re.search(line)
            if t_match:
                data["epoch"].append(current_epoch)
                data["train_total"].append(float(t_match.group(1)))
                data["train_mag"].append(float(t_match.group(2)))
                data["train_phase"].append(float(t_match.group(3)))
            
            v_match = val_re.search(line)
            if v_match:
                data["val_total"].append(float(v_match.group(1)))
                data["val_mag"].append(float(v_match.group(2)))
                data["val_phase"].append(float(v_match.group(3)))
    
    return data

def plot_metrics(data, log_path, custom_save_path):
    # 逻辑处理：如果未指定 save_path，则设为 log 所在文件夹
    if custom_save_path is None:
        log_dir = os.path.dirname(os.path.abspath(log_path))
        save_path = os.path.join(log_dir, "loss_curve.png")
    else:
        save_path = custom_save_path

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Metrics - {os.path.basename(log_dir)}", fontsize=16)
    
    metrics = [
        ('Total Loss', 'train_total', 'val_total'),
        ('Mag Loss', 'train_mag', 'val_mag'),
        ('Phase Loss', 'train_phase', 'val_phase')
    ]
    
    for i, (title, t_key, v_key) in enumerate(metrics):
        axes[i].plot(data["epoch"], data[t_key], label='Train', color='#1f77b4')
        axes[i].plot(data["epoch"], data[v_key], label='Val', color='#ff7f0e', linestyle='--')
        axes[i].set_title(title)
        axes[i].set_xlabel('Epoch')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"Successfully plotted {len(data['epoch'])} epochs.")
    print(f"Plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    log_data = extract_metrics(args.log_path)
    if log_data and log_data["epoch"]:
        plot_metrics(log_data, args.log_path, args.save_path)