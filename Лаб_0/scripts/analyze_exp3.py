import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

def main():
    run_dirs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])
    
    # 收集最后 10 次带 eval-test 的实验数据
    data = []
    for d in run_dirs:
        try:
            with open(d / "metrics.json", "r") as f:
                metrics = json.load(f)
            
            # 必须包含 test_acc 才说明是实验 3 的数据
            if "test_acc" not in metrics:
                continue
                
            parts = d.name.split("_")
            seed = int(parts[-1].replace("seed", ""))
            
            # 读 config 来判断是 base 还是 best
            import yaml
            with open(d / "config.yaml", "r") as f:
                config = yaml.safe_load(f)
                experiment = config.get("experiment", "")
                # 如果是 base.yaml 跑的，experiment 叫 lr0；如果是 best.yaml 跑的，experiment 叫 lr0_best
                config_type = "best" if "best" in experiment else "base"
                
            preds = pd.read_csv(d / "predictions.csv")
            test_preds = preds[preds["split"] == "test"].copy()
            test_preds["correct"] = test_preds["y_true"] == test_preds["y_pred"]
            test_pred_dict = test_preds.set_index("index")["correct"].to_dict()

            data.append({
                "seed": seed,
                "config": config_type,
                "val_acc": metrics["val_acc"],
                "test_acc": metrics["test_acc"],
                "test_preds": test_pred_dict
            })
        except Exception:
            continue

    df = pd.DataFrame(data)
    if df.empty:
        print("未找到实验 3 的数据（可能是忘了加 --eval-test）")
        return

    # 按照 seed 分组对比
    print("=" * 60)
    print("实验 3：基础配置 vs 最佳配置")
    print("=" * 60)
    
    base_df = df[df["config"] == "base"].set_index("seed").sort_index()
    best_df = df[df["config"] == "best"].set_index("seed").sort_index()
    
    # 确保 101-105 都存在
    seeds = [101, 102, 103, 104, 105]
    
    print(f"{'Seed':<6} | {'Base Val':<9} | {'Best Val':<9} | {'Δ Val':<8} | {'Base Test':<9} | {'Best Test':<9} | {'Δ Test':<8}")
    print("-" * 72)
    
    delta_vals = []
    delta_tests = []
    fixed_images = 0
    broken_images = 0

    for s in seeds:
        if s not in base_df.index or s not in best_df.index:
            continue
            
        b_val = base_df.loc[s, "val_acc"]
        m_val = best_df.loc[s, "val_acc"]
        b_test = base_df.loc[s, "test_acc"]
        m_test = best_df.loc[s, "test_acc"]
        
        d_val = m_val - b_val
        d_test = m_test - b_test
        
        delta_vals.append(d_val)
        delta_tests.append(d_test)
        
        # 统计 test 集上改对/改错的数量
        b_preds = base_df.loc[s, "test_preds"]
        m_preds = best_df.loc[s, "test_preds"]
        
        for img_idx in b_preds.keys():
            if b_preds[img_idx] == False and m_preds[img_idx] == True:
                fixed_images += 1
            elif b_preds[img_idx] == True and m_preds[img_idx] == False:
                broken_images += 1
        
        print(f"{s:<6} | {b_val:.4f}    | {m_val:.4f}    | {d_val:+.4f}  | {b_test:.4f}    | {m_test:.4f}    | {d_test:+.4f}")

    print("-" * 72)
    print(f"Δ Test 中位数: {pd.Series(delta_tests).median():+.4f}")
    print(f"Δ Test 范围:  [{min(delta_tests):+.4f}, {max(delta_tests):+.4f}]")
    print(f"在 test 集上，最佳配置共改对了 {fixed_images} 张图，改错了 {broken_images} 张图。")
    print("=" * 60)

    # 回答问题：三个数对比
    print("\n对比三个数（验证 selection bias）：")
    print(f"1. 搜索中最优 trial 的 Val Acc: 0.9833 (之前终端打印的结果)")
    print(f"2. 最佳配置在新 seed 下的平均 Val Acc: {best_df['val_acc'].mean():.4f}")
    print(f"3. 最佳配置在新 seed 下的平均 Test Acc: {best_df['test_acc'].mean():.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()