import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = f"sqlite:///{PROJECT_ROOT}/optuna.db"

def main():
    # 加载 study
    study = optuna.load_study(study_name="lr0_search", storage=DB_PATH)
    trials = study.trials

    # 1. 统计状态
    states = [t.state.name for t in trials]
    state_counts = Counter(states)
    print("=" * 50)
    print("实验 2：Optuna 搜索统计")
    print("=" * 50)
    print(f"总计 Trial 数: {len(trials)}")
    for state in ["COMPLETE", "PRUNED", "FAIL"]:
        print(f"  {state}: {state_counts.get(state, 0)}")

    # 2. 统计实际执行的 Epoch 数
    actual_epochs = sum(len(t.intermediate_values) for t in trials if t.intermediate_values)
    theoretical_epochs = len(trials) * 30
    print(f"\n理论应跑 Epoch 数: {theoretical_epochs}")
    print(f"实际跑了 Epoch 数: {actual_epochs}")
    print(f"Pruner 节省了约: {theoretical_epochs - actual_epochs} 个 Epoch 的计算时间")
    print("=" * 50)

    # 3. 画图：优化历史（Trial 编号 vs 最佳值）
    plt.figure(figsize=(10, 5))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "optimization_history.png", dpi=150)
    print("\n已保存图表: optimization_history.png")

    # 4. 画图：参数重要性
    try:
        plt.figure(figsize=(10, 5))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title("Hyperparameter Importances")
        plt.tight_layout()
        plt.savefig(PROJECT_ROOT / "param_importances.png", dpi=150)
        print("已保存图表: param_importances.png")
    except Exception as e:
        print(f"无法生成参数重要性图 (可能是完成的 trial 太少): {e}")

if __name__ == "__main__":
    main()