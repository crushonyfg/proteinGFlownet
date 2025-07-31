from datasets import load_dataset

# 选你需要的子集，比如 "dataset3_single"
dataset = load_dataset("RosettaCommons/MegaScale", name="dataset3_single")

filename = "data/MegaScale"
import os
os.makedirs(filename, exist_ok=True)

# 查看已有 split
print(dataset)

# 把 train/val/test 分别保存为 parquet（也可以 to_csv，但 parquet 更节省空间和速度）
dataset["train"].to_parquet(f"{filename}/dataset3_single_train.parquet")
dataset["val"].to_parquet(f"{filename}/dataset3_single_val.parquet")
dataset["test"].to_parquet(f"{filename}/dataset3_single_test.parquet")
