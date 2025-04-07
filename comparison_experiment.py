from run_experiment import run_experiment
import itertools
import json
from tqdm import tqdm

# 激活函数和优化器的组合
activation_functions = ["relu", "tanh", "leaky_relu"]
optimizers = ["sgd", "adam"]

# 固定参数
fixed_config = {
    "learning_rate": 1e-3,
    "momentum": 0.9,
    "dropout_prob": 0.2,
    "hidden_dims": [512, 256, 128],
    "batch_size": 64,
    "weight_decay": 0.0005,
    "num_epochs": 50,
    "input_dim": 128,
    "output_dim": 10,
    "use_batchnorm": True,
    "early_stop_patience": 5,
    "early_stop_threshold": 0.005
}

# 组合所有激活函数和优化器
grid = list(itertools.product(activation_functions, optimizers))
print(f"Total comparison experiments planned: {len(grid)}")

results = []

for i, (act_fn, opt_name) in enumerate(tqdm(grid, desc="Running Comparison Experiments")):
    config = fixed_config.copy()
    config["activation"] = act_fn
    config["optimizer"] = opt_name

    print(f"\n=== Running Experiment {i} ===")
    print(json.dumps(config, indent=2))

    try:
        result = run_experiment(config, trial_id=f"cmp{i}")
        result["config"] = config
        result["exp_id"] = f"cmp{i}"
        results.append(result)
    except Exception as e:
        print(f"Experiment cmp{i} failed: {e}")
        continue

    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)

# 输出 Top 3
top = sorted(results, key=lambda x: x["f1"], reverse=True)[:3]

print("\nTop 3 Comparison Results:")
for i, r in enumerate(top, 1):
    print(f"\nTop {i}: F1 = {r['f1']:.4f}, Acc = {r['test_acc']:.4f}")
    print(json.dumps(r["config"], indent=2))
