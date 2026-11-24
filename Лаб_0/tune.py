import optuna
import mlflow
import yaml
import json
import time
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# 从我们写好的 train.py 中导入组件
from train import MLP, get_dataloaders, evaluate, set_seed, flatten_config

PROJECT_ROOT = Path(__file__).resolve().parent

def objective(trial):
    # 1. 定义搜索空间
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden = trial.suggest_int("hidden", 32, 512, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # 固定参数
    seed = 0
    epochs = 30
    
    # 构造配置
    config = {
        "experiment": "lr0_search",
        "seed": seed,
        "split": {"path": "data/split.json"},
        "model": {"n_layers": n_layers, "hidden": hidden, "dropout": dropout},
        "train": {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size, "epochs": epochs},
        "runtime": {"device": "cpu", "num_threads": 1}
    }
    
    torch.set_num_threads(1)
    set_seed(seed)
    
    # 嵌套的 MLflow 记录
    with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
        mlflow.log_params(flatten_config(config))
        
        try:
            train_loader, val_loader, test_loader, _, _ = get_dataloaders(config, seed)
            
            model = MLP(
                input_dim=64,
                n_layers=n_layers,
                hidden=hidden,
                dropout=dropout
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                for X_batch, y_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * X_batch.size(0)
                    
                train_loss /= len(train_loader.dataset)
                _, val_acc, _, _ = evaluate(model, val_loader)
                
                # 记录逐 epoch 的指标
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                
                # 报告给 Optuna 的剪枝器
                trial.report(val_acc, epoch)
                
                # 检查是否应该剪枝
                if trial.should_prune():
                    mlflow.set_tag("status", "pruned")
                    raise optuna.TrialPruned()
                    
            mlflow.log_metric("final_val_acc", val_acc)
            mlflow.set_tag("status", "completed")
            return val_acc
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            mlflow.set_tag("status", "failed")
            print(f"Trial {trial.number} failed with error: {e}")
            return float('nan')

def main():
    # 设置 SQLite 存储
    storage_url = "sqlite:///optuna.db"
    study_name = "lr0_search"
    
    # 创建采样器和剪枝器
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    # 创建或加载 study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )
    
    # 设置 MLflow 父运行
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("lr0_search_experiment")
    
    with mlflow.start_run(run_name="Optuna_Search_Parent"):
        # 开始优化
        print("Starting Optuna optimization (40 trials)...")
        study.optimize(objective, n_trials=40)
        
        # 打印最佳结果
        print("\nOptimization finished!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value (val_acc): {study.best_value:.4f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()