import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

def main():
    # 获取所有带有 seed 的 run 文件夹
    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])
    
    results = []
    predictions = {}
    
    # 找到 seed 1 到 5 的最新运行
    latest_seeds = {}
    for d in run_dirs:
        try:
            # 从文件夹名解析 seed，例如 20231117_150622_seed0
            parts = d.name.split("_")
            seed = int(parts[-1].replace("seed", ""))
            # 因为我们按时间戳排序，覆盖更新之前的
            latest_seeds[seed] = d
        except ValueError:
            continue

    print("=" * 60)
    print("实验 1 汇总：Seed 波动")
    print("=" * 60)
    
    val_accs = []
    
    for seed in range(1, 6):
        if seed not in latest_seeds:
            print(f"[WARN] 未找到 seed {seed} 的运行结果！")
            continue
            
        d = latest_seeds[seed]
        
        # 读取 metrics
        with open(d / "metrics.json", "r") as f:
            metrics = json.load(f)
            
        # 读取 history 拿最后一个 epoch 的 time 或直接算总时间
        with open(d / "history.csv", "r") as f:
            history = pd.read_csv(f)
            epoch_times = history["epoch_time"].sum()
            
        val_accs.append(metrics["val_acc"])
        
        # 读取 predictions 用于对比
        preds_df = pd.read_csv(d / "predictions.csv")
        # 只取 val 部分
        preds_val = preds_df[preds_df["split"] == "val"].copy()
        # 记录预测正确的
        preds_val["correct"] = preds_val["y_true"] == preds_val["y_pred"]
        predictions[seed] = preds_val.set_index("index")["correct"]
        
        results.append({
            "Seed": seed,
            "Val Acc": f"{metrics['val_acc']:.4f}",
            "Train Time (s)": f"{metrics['train_time_sec']:.2f}",
            "Epoch Time Sum (s)": f"{epoch_times:.2f}"
        })
        
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    print("-" * 60)
    if val_accs:
        print(f"Min Val Acc:    {min(val_accs):.4f}")
        print(f"Median Val Acc: {pd.Series(val_accs).median():.4f}")
        print(f"Max Val Acc:    {max(val_accs):.4f}")
    print("=" * 60)

    # 对比两个不同 seed 的预测差异（例如 seed 1 和 seed 2）
    if 1 in predictions and 2 in predictions:
        print("\n预测差异对比 (Seed 1 vs Seed 2):")
        p1 = predictions[1]
        p2 = predictions[2]
        
        # 对齐 index
        joined = pd.DataFrame({"s1": p1, "s2": p2}).dropna()
        
        s1_only = joined[(joined["s1"] == True) & (joined["s2"] == False)]
        s2_only = joined[(joined["s1"] == False) & (joined["s2"] == True)]
        both_wrong = joined[(joined["s1"] == False) & (joined["s2"] == False)]
        both_right = joined[(joined["s1"] == True) & (joined["s2"] == True)]
        
        print(f"两个 seed 都正确: {len(both_right)} 张图片")
        print(f"两个 seed 都错误: {len(both_wrong)} 张图片")
        print(f"仅 seed 1 正确:   {len(s1_only)} 张图片")
        print(f"仅 seed 2 正确:   {len(s2_only)} 张图片")

if __name__ == "__main__":
    main()