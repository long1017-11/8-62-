import argparse
import json
import random
import time
import datetime
import yaml
import subprocess
import traceback
import sys
import platform
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_digits
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================
# 1. 工具函数
# ==========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def check_git_status():
    """检查是否有未提交的代码修改。如果有，拒绝运行。"""
    try:
        # 检查是否有未提交的改动（包括 untracked 文件）
        status = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip()
        if status:
            print("[ERROR] 存在未提交的代码修改，拒绝运行实验！")
            print("修改的文件如下：")
            print(status)
            print("\n请先执行 git add . && git commit -m 'update' 后再试。")
            sys.exit(1)
    except FileNotFoundError:
        print("[WARN] 未检测到 Git，跳过未提交代码检查。")
    except subprocess.CalledProcessError:
        print("[WARN] 当前目录不是 Git 仓库，跳过检查。")

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    except Exception:
        return "unknown"

def get_package_version(pkg):
    try:
        return importlib.metadata.version(pkg)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

def flatten_config(d, parent_key='', sep='.'):
    """将嵌套的 yaml 配置展平，用于 MLflow 记录"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# ==========================================
# 2. 模型定义
# ==========================================
class MLP(nn.Module):
    def __init__(self, input_dim, n_layers, hidden, dropout, num_classes=10):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 数据准备
# ==========================================
def get_dataloaders(config, seed):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    split_path = PROJECT_ROOT / config["split"]["path"]
    with open(split_path, "r") as f:
        split_idx = json.load(f)

    train_idx, val_idx, test_idx = split_idx["train"], split_idx["val"], split_idx["test"]

    train_dataset = TensorDataset(X_tensor[train_idx], y_tensor[train_idx])
    val_dataset = TensorDataset(X_tensor[val_idx], y_tensor[val_idx])
    test_dataset = TensorDataset(X_tensor[test_idx], y_tensor[test_idx])

    g = torch.Generator()
    g.manual_seed(seed)

    batch_size = config["train"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, val_idx, test_idx

# ==========================================
# 4. 评估函数
# ==========================================
def evaluate(model, loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * X_batch.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc, all_preds, all_targets

# ==========================================
# 5. 主训练流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    # 0. 检查 Git 状态
    check_git_status()

    # 1. 读取配置
    config_path = PROJECT_ROOT / args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["seed"] = args.seed
    seed = config["seed"]
    epochs = config["train"]["epochs"]

    torch.set_num_threads(config["runtime"]["num_threads"])
    set_seed(seed)

    # 2. 生成目录和基础信息
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_seed{seed}"
    run_dir = PROJECT_ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    start_time_str = datetime.datetime.now().isoformat()
    commit_hash = get_git_commit()

    # 读取 split hash
    import hashlib
    split_path = PROJECT_ROOT / config["split"]["path"]
    split_hash = hashlib.sha256(split_path.read_bytes()).hexdigest()

    metadata = {
        "run_id": run_id,
        "series": config.get("experiment", "default"),
        "status": "running",
        "start_time": start_time_str,
        "end_time": None,
        "commit": commit_hash,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "sklearn_version": get_package_version("scikit-learn"),
        "optuna_version": get_package_version("optuna"),
        "mlflow_version": get_package_version("mlflow"),
        "num_threads": config["runtime"]["num_threads"],
        "seed": seed,
        "split_sha256": split_hash
    }
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. 设置 MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0_experiment")

    with mlflow.start_run(run_name=run_id) as active_run:
        # 记录参数和标签
        mlflow.log_params(flatten_config(config))
        mlflow.set_tags({
            "commit": commit_hash,
            "series": config.get("experiment", "default"),
            "seed": str(seed)
        })

        history = []
        try:
            train_loader, val_loader, test_loader, val_idx, test_idx = get_dataloaders(config, seed)

            model = MLP(
                input_dim=64,
                n_layers=config["model"]["n_layers"],
                hidden=config["model"]["hidden"],
                dropout=config["model"]["dropout"]
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=config["train"]["lr"], 
                weight_decay=config["train"]["weight_decay"]
            )

            print(f"开始运行: {run_id}")
            train_start = time.time()

            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                epoch_start = time.time()
                
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                    
                train_loss /= len(train_loader.dataset)
                val_loss, val_acc, _, _ = evaluate(model, val_loader)
                epoch_time = time.time() - epoch_start
                
                history.append({
                    "epoch": epoch, "train_loss": train_loss, 
                    "val_loss": val_loss, "val_acc": val_acc, "epoch_time": epoch_time
                })
                
                # 记录逐 epoch 的 MLflow 指标
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)

                print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            total_time = time.time() - train_start

            # 保存历史
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            # 最终预测和保存
            _, final_val_acc, val_preds, val_targets = evaluate(model, val_loader)
            preds_data = []
            for i, idx in enumerate(val_idx):
                preds_data.append({"split": "val", "index": idx, "y_true": val_targets[i], "y_pred": val_preds[i]})
                
            test_acc = None
            if args.eval_test:
                _, test_acc, test_preds, test_targets = evaluate(model, test_loader)
                for i, idx in enumerate(test_idx):
                    preds_data.append({"split": "test", "index": idx, "y_true": test_targets[i], "y_pred": test_preds[i]})
                    
            pd.DataFrame(preds_data).to_csv(run_dir / "predictions.csv", index=False)

            # 最终指标
            metrics = {"val_acc": final_val_acc, "val_loss": history[-1]["val_loss"], "train_time_sec": total_time}
            if test_acc is not None:
                metrics["test_acc"] = test_acc
                
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # 记录最终 MLflow 指标
            mlflow.log_metric("final_val_acc", final_val_acc)
            if test_acc is not None:
                mlflow.log_metric("final_test_acc", test_acc)
            
            # 记录 artifact（整个 run 目录）
            mlflow.log_artifacts(run_dir, artifact_path="run_files")

            # 更新 metadata 为 completed
            metadata["status"] = "completed"
            metadata["end_time"] = datetime.datetime.now().isoformat()
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"[SUCCESS] 运行结束！Val Acc: {final_val_acc:.4f}")
            if test_acc is not None:
                print(f"   Test Acc: {test_acc:.4f}")

        except Exception as e:
            print(f"[FAILED] 运行失败: {e}")
            traceback.print_exc()
            
            # 保存失败状态和错误日志
            metadata["status"] = "failed"
            metadata["end_time"] = datetime.datetime.now().isoformat()
            with open(run_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            with open(run_dir / "error.log", "w") as f:
                f.write(traceback.format_exc())
                
            mlflow.set_tag("status", "failed")
            mlflow.log_artifact(run_dir / "error.log")
            sys.exit(1)

if __name__ == "__main__":
    main()