import os, torch, random, time, json, glob
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler
from models import NaturalResourceFoundationModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def save_visuals(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1); plt.plot(history["loss"], color='#2ca02c'); plt.title('ZRZY-v1 (0.1B) Loss'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2); plt.plot(history["lr"], color='#1f77b4'); plt.title('LR Schedule (1000 Epochs)'); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig("training_monitor.png", dpi=150); plt.close()

def load_all_caches(cache_files, is_master):
    all_layers_data = {}
    for file in cache_files:
        if os.path.exists(file):
            if is_master: print(f"📦 装载高速缓存: {file}")
            data = torch.load(file, map_location='cpu', weights_only=False)
            all_layers_data.update(data)
    return all_layers_data

def train():
    local_rank = setup_ddp(); is_master = (dist.get_rank() == 0)
    
    config = {'vocab_size': 20000}
    batch_size, epochs, base_lr, warmup_epochs = 1024, 1000, 1.2e-4, 50
    
    model = DDP(NaturalResourceFoundationModel(config).cuda(local_rank), device_ids=[local_rank], find_unused_parameters=False)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    scaler = GradScaler('cuda')

    cache_files = glob.glob("cache_*.pt")
    if is_master: 
        print(f"🧲 共发现 {len(cache_files)} 个张量缓存文件，准备汇入训练池...")

    data_all = load_all_caches(cache_files, is_master)
    layer_names = list(data_all.keys())
    history = {"loss": [], "lr": []}; best_loss = float('inf'); start_time = time.time()

    for epoch in range(epochs):
        model.train(); epoch_loss, total_b = 0.0, 0
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups: pg['lr'] = lr
        
        random.shuffle(layer_names)
        for ln in layer_names:
            d = data_all[ln]
            if d['cont_int'].shape[0] < 2: continue
            
            ds = TensorDataset(*[torch.from_numpy(d[k]) for k in ['cont_int','cont_frac_hi','cont_frac_lo','cont_norm','word_data','char_data']])
            sm = DistributedSampler(ds, shuffle=True); sm.set_epoch(epoch)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sm, num_workers=24, pin_memory=True, persistent_workers=True)
            
            for batch in loader:
                batch = [t.cuda(local_rank, non_blocking=True) for t in batch]
                optimizer.zero_grad()
                with autocast('cuda'):
                    _, loss = model(*batch); loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                epoch_loss += loss.item(); total_b += 1

        if epoch >= warmup_epochs: scheduler.step()
        if is_master and total_b > 0:
            avg_loss = epoch_loss / total_b; curr_lr = optimizer.param_groups[0]['lr']
            history["loss"].append(avg_loss); history["lr"].append(curr_lr)
            if (epoch + 1) % 5 == 0:
                save_visuals(history)
                with open("train_history.json", "w") as f: json.dump(history, f)
            if avg_loss < best_loss:
                best_loss = avg_loss; 
                torch.save(model.module.state_dict(), "best_model.pth")
            print(f"📈 Ep {epoch+1:04d}/{epochs} | Loss: {avg_loss:.6f} | LR: {curr_lr:.2e} | T: {time.time()-start_time:.1f}s")

    dist.destroy_process_group()

if __name__ == "__main__": train()