# grid_search.py

from run_experiment import run_experiment
import itertools
import json
from tqdm import tqdm

# 超参数搜索空间，避免过拟合：只尝试轻度 Dropout 和高动量组合
learning_rates = [1e-4, 1e-3]
momentums = [0.0, 0.9]  # 精简搜索空间
dropout_probs = [0.0, 0.2]  # 加入正则化尝试
hidden_dims_list = [[512, 256, 128]]
batch_sizes = [64]

# 固定参数
fixed_config = {
    "num_epochs": 100,
    "weight_decay": 0.0005,
    "input_dim": 128,
    "output_dim": 10
}

# 组合所有参数
grid = list(itertools.product(learning_rates, momentums, dropout_probs, hidden_dims_list, batch_sizes))
print(f"Total experiments planned: {len(grid)}")

results = []

# 运行网格搜索: 使用tqdm 进度条库，来在长时间运行的循环中显示进度
for i, (lr, mo, dr, hd, bs) in enumerate(tqdm(grid, desc="Running Experiments")):
    config = {
        **fixed_config,
        "learning_rate": lr,
        "momentum": mo,
        "dropout_prob": dr,
        "hidden_dims": hd,
        "batch_size": bs,
        "early_stop_patience": 5,     # 支持 Early Stopping：若 5 轮无提升提前停止
        "early_stop_threshold": 0.01  # 损失或 acc 改变不超过 1% 则提前停止
    }

    print(f"\n=== Running Experiment {i} ===")
    print(json.dumps(config, indent=2))

    try:
        result = run_experiment(config, trial_id=f"exp{i}")
        result["config"] = config
        result["exp_id"] = i
        results.append(result)
    except Exception as e:
        print(f"Experiment {i} failed with error: {e}")
        continue

    # 实时保存中间结果
    with open("grid_search_results.json", "w") as f:
        json.dump(results, f, indent=4)

# 最终打印 Top 3 结果
top = sorted(results, key=lambda x: x["f1"], reverse=True)[:3]

print("\nTop 3 Configurations by F1 Score:")
for i, r in enumerate(top, 1):
    print(f"\nTop {i}: F1 = {r['f1']:.4f}, Test Acc = {r['test_acc']:.4f}")
    print(json.dumps(r["config"], indent=2))

print("\nAll experiments completed. Results saved to grid_search_results.json.")

# 加载实验结果
with open("grid_search_results.json", "r") as f:
    results = json.load(f)

# 找出 F1 分数最高的模型
best_result = max(results, key=lambda x: x["f1"])
best_id = best_result["exp_id"]
best_f1 = best_result["f1"]
best_acc = best_result["test_acc"]
best_config = best_result["config"]

print(f"最佳模型是：Experiment {best_id}")
print(f"Test Accuracy: {best_acc:.4f} | F1 Score: {best_f1:.4f}")
print("模型配置如下：")
print(best_config)

print(f"模型权重文件：saved_models/best_model_exp{best_id}.pkl")
print(f"训练曲线图文件：saved_models/train_curve_exp{best_id}.png")
