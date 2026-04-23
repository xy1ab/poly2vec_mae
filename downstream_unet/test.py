#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py
@Time    :   2026/04/23 16:59:37
@Author  :   Hu Bin 
@Version :   1.0
@Desc    :   None
'''


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pt2tar.py
@Time    :   2026/04/22 09:56:27
@Author  :   Hu Bin
@Version :   1.0
@Desc    :   Convert pt shards to WebDataset tar shards with CPU multiprocessing.
'''

import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import webdataset as wds
from tqdm import tqdm

def _value_nbytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    return np.asarray(value).nbytes


def _sample_nbytes(sample_dict):
    return _value_nbytes(sample_dict["input.npy"]) + _value_nbytes(sample_dict["label.npy"])


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _write_tar_task(task):
    output_path = task["output_path"]
    samples = task["samples"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with wds.TarWriter(output_path) as sink:
        for sample in samples:
            sink.write(sample)

    return {
        "split": task["split"],
        "path": os.path.abspath(output_path),
        "num_samples": len(samples),
        "payload_bytes": task["payload_bytes"],
        "first_key": samples[0]["__key__"] if samples else None,
        "last_key": samples[-1]["__key__"] if samples else None,
    }


def _check_output_dir(output_dir):
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    existing = []
    existing.extend(glob.glob(os.path.join(train_dir, "train-*.tar")))
    existing.extend(glob.glob(os.path.join(val_dir, "val-*.tar")))
    if existing:
        preview = "\n".join(sorted(existing)[:5])
        raise FileExistsError(f"输出目录已存在 tar 文件，请更换目录或手动清理后重试:\n{preview}")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    return train_dir, val_dir


def _make_sample(pt_file, sample_index, sample):
    key = f"{os.path.basename(pt_file).replace('.pt', '')}_{sample_index:06d}"
    return {
        "__key__": key,
        "input.npy": _to_numpy(sample["icft"])[np.newaxis, ...],
        "label.npy": _to_numpy(sample["rec_label"])[np.newaxis, ...],
    }


def _finalize_bucket(split, output_dir, shard_index, bucket, payload_bytes):
    filename = f"{split}-{shard_index:06d}.tar"
    return {
        "split": split,
        "output_path": os.path.join(output_dir, split, filename),
        "samples": bucket,
        "payload_bytes": payload_bytes,
    }


def _add_to_bucket(split, sample_dict, sample_bytes, buckets, task_lists, output_dir, shard_indices, max_size_bytes):
    bucket = buckets[split]["samples"]
    payload_bytes = buckets[split]["payload_bytes"]

    if bucket and payload_bytes + sample_bytes > max_size_bytes:
        task_lists[split].append(
            _finalize_bucket(split, output_dir, shard_indices[split], bucket, payload_bytes)
        )
        shard_indices[split] += 1
        bucket = []
        payload_bytes = 0

    bucket.append(sample_dict)
    payload_bytes += sample_bytes
    buckets[split] = {"samples": bucket, "payload_bytes": payload_bytes}


def _submit_and_collect(executor, tasks, write_pbar):
    if not tasks:
        return []

    results = []
    futures = [executor.submit(_write_tar_task, task) for task in tasks]
    for future in as_completed(futures):
        result = future.result()
        results.append(result)
        write_pbar.update(result["num_samples"])
    return results


def convert_pt_to_tar(
    input_dir,
    output_dir,
    train_ratio=0.9,
    max_size_bytes=1 * 1024**3,
    pt_batch_size=4,
    num_workers=None,
):
    """
    input_dir: 包含 .pt 文件的文件夹
    output_dir: .tar 文件输出目录
    max_size_bytes: 每个 .tar 包的大小上限
    pt_batch_size: 每轮主进程一次性加载的 .pt 文件数
    num_workers: 并行写 tar 的 CPU worker 数
    """
    if pt_batch_size < 1:
        raise ValueError("pt_batch_size 必须 >= 1")
    if max_size_bytes < 1:
        raise ValueError("max_size_bytes 必须 >= 1")

    num_workers = int(num_workers or min(4, os.cpu_count() or 1))
    if num_workers < 1:
        raise ValueError("num_workers 必须 >= 1")

    train_dir, val_dir = _check_output_dir(output_dir)
    pt_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"未找到 .pt 文件: {input_dir}")

    threshold = train_ratio * 100
    buckets = {
        "train": {"samples": [], "payload_bytes": 0},
        "val": {"samples": [], "payload_bytes": 0},
    }
    shard_indices = {"train": 0, "val": 0}
    split_counts = {"train": 0, "val": 0}
    total_samples = 0
    input_files = []
    tar_records = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=len(pt_files), desc="Loading pt files", unit="file") as load_pbar:
            with tqdm(desc="Writing samples", unit="sample") as write_pbar:
                for start in range(0, len(pt_files), pt_batch_size):
                    batch_files = pt_files[start:start + pt_batch_size]
                    task_lists = {"train": [], "val": []}

                    for pt_file in batch_files:
                        data_list = torch.load(pt_file, map_location="cpu", weights_only=False)
                        input_files.append({
                            "path": os.path.abspath(pt_file),
                            "num_samples": len(data_list),
                        })

                        for sample_index, sample in enumerate(data_list):
                            sample_dict = _make_sample(pt_file, sample_index, sample)
                            sample_bytes = _sample_nbytes(sample_dict)
                            split = "train" if (total_samples % 100) < threshold else "val"

                            _add_to_bucket(
                                split,
                                sample_dict,
                                sample_bytes,
                                buckets,
                                task_lists,
                                output_dir,
                                shard_indices,
                                max_size_bytes,
                            )
                            split_counts[split] += 1
                            total_samples += 1

                        del data_list
                        load_pbar.update(1)

                    tasks = task_lists["train"] + task_lists["val"]
                    tar_records.extend(_submit_and_collect(executor, tasks, write_pbar))

                tail_tasks = []
                for split in ("train", "val"):
                    bucket = buckets[split]["samples"]
                    payload_bytes = buckets[split]["payload_bytes"]
                    if bucket:
                        tail_tasks.append(
                            _finalize_bucket(split, output_dir, shard_indices[split], bucket, payload_bytes)
                        )
                        shard_indices[split] += 1
                        buckets[split] = {"samples": [], "payload_bytes": 0}

                tar_records.extend(_submit_and_collect(executor, tail_tasks, write_pbar))

    index_data = {
        "train": {
            "num_samples": split_counts["train"],
            "path": os.path.abspath(train_dir),
        },
        "val": {
            "num_samples": split_counts["val"],
            "path": os.path.abspath(val_dir),
        },
    }
    index_path = os.path.join(output_dir, "index_file.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index_data, f, indent=4)

    manifest = {
        "input_dir": os.path.abspath(input_dir),
        "output_dir": os.path.abspath(output_dir),
        "pt_batch_size": pt_batch_size,
        "num_workers": num_workers,
        "max_size_bytes": max_size_bytes,
        "train_ratio": train_ratio,
        "total_samples": total_samples,
        "train_samples": split_counts["train"],
        "val_samples": split_counts["val"],
        "input_files": input_files,
        "tar_files": sorted(tar_records, key=lambda item: item["path"]),
    }
    manifest_path = os.path.join(output_dir, "pt2tar_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4)

    print("转换完成。")
    print(f"总样本数: {total_samples}")
    print(f"训练集样本数: {split_counts['train']}, 验证集样本数: {split_counts['val']}")
    print(f"索引文件已保存至: {index_path}")
    print(f"Manifest 已保存至: {manifest_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Convert pt files to WebDataset tar shards with CPU multiprocessing.")
    parser.add_argument("--input_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset")
    parser.add_argument("--output_dir", type=str, default="/mnt/git-data/HB/poly2vec_mae/outputs/unet_traindataset")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_size_gb", type=float, default=1.0)
    parser.add_argument("--max_size_bytes", type=int, default=None)
    parser.add_argument("--pt_batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count() or 1))
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    max_size_bytes = args.max_size_bytes
    if max_size_bytes is None:
        max_size_bytes = int(args.max_size_gb * 1024**3)

    convert_pt_to_tar(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        max_size_bytes=max_size_bytes,
        pt_batch_size=args.pt_batch_size,
        num_workers=args.num_workers,
    )

