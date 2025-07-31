#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm

def save_pdb(name: str, pdb_field: str, out_dir: Path):
    out_path = out_dir / name
    if out_path.exists():
        return  # 已存在跳过
    # 粗略判断：如果看起来像是 PDB 文件内容（含有ATOM/HETATM行），就直接写
    if ("ATOM" in pdb_field or "HETATM" in pdb_field) and "\n" in pdb_field:
        out_path.write_text(pdb_field)
    else:
        # 否则当作 URL 下载
        try:
            resp = requests.get(pdb_field, timeout=30)
            resp.raise_for_status()
            out_path.write_bytes(resp.content)
        except Exception as e:
            print(f"[WARN] 下载 {name} 失败，从 {pdb_field} : {e}")

def main():
    parser = argparse.ArgumentParser(description="下载 AlphaFold_model_PDBs 中的 PDB 到本地")
    parser.add_argument(
        "--out_dir", "-o", type=Path, default="data/MegaScale/AlphaFold_model_PDBs",
        help="保存 PDB 的本地目录"
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="最多下载多少个（调试用）"
    )
    parser.add_argument(
        "--cache_dir", type=Path, default=None,
        help="可选的 HuggingFace 缓存目录"
    )
    args = parser.parse_args()

    ds = load_dataset(
        "RosettaCommons/MegaScale",
        name="AlphaFold_model_PDBs",
        data_dir="AlphaFold_model_PDBs",
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )["train"]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = len(ds) if args.limit is None else min(len(ds), args.limit)
    for i in tqdm(range(total), desc="Downloading PDBs"):
        item = ds[i]
        name = item.get("name")  # 文件名，通常形如 something.pdb
        pdb_field = item.get("pdb")  # 可能是文本也可能是 URL
        if name is None or pdb_field is None:
            continue
        save_pdb(name, pdb_field, args.out_dir)

if __name__ == "__main__":
    main()
