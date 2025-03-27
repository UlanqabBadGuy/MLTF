import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import os

# === 导入模型封装类 === #
from LightGBM import LightGBMLoanModel
from LReg import LogisticRegressionLoanModel
from MLPTrainer import MLPTrainer
from XGBoost import XGBoostLoanModel
from CatBoost import CatBoostLoanModel

# === 加载和划分数据 === #
df = pd.read_csv("../data/Loan_Default/Loan_Default_Preprocessed.csv")
target_var = ['Status']
X = df.drop(columns=target_var)
y = df[target_var]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 模型配置和调参 === #
model_configs = {
    "CatBoost": {
        "class": CatBoostLoanModel,
        "params_grid": {
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8]
        }
    },
    "LightGBM": {
        "class": LightGBMLoanModel,
        "params_grid": {
            'num_leaves': [31, 64, 128],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [5, 10, -1]
        }
    },
    "LogisticRegression": {
        "class": LogisticRegressionLoanModel,
        "params_grid": {
            'C': [0.01, 0.1, 1.0, 10.0]
        }
    },
    "XGBoost": {
        "class": XGBoostLoanModel,
        "params_grid": {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 6, 10],
            'n_estimators': [100, 200]
        }
    },
    "MLP": {
        "class": MLPTrainer,
        "params_grid": {
            'batch_size': [32, 64],
            'lr': [1e-3, 1e-4],
            'epochs': [10, 15]
        }
    }
}

results = []
all_trials = []
os.makedirs("saved_models", exist_ok=True)

# ========== 启动调参与模型对比 ========== #
for model_name, config in model_configs.items():
    print(f"\n=== 🔍 Tuning & Evaluating {model_name} ===")

    best_score = -np.inf
    best_model = None
    best_params = None

    from itertools import product
    keys, values = zip(*config['params_grid'].items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]

    for params in param_combinations:
        print(f"\n🔧 Trying params: {params}")

        cls = config['class']
        if model_name == "MLP":
            model = cls(X_train, y_train, X_test, y_test, **params)
            model.train()
            metrics = model.evaluate(return_metrics=True)
        else:
            model = cls(params=params)
            model.fit(X_train.copy(), y_train)
            metrics = model.evaluate(X_test.copy(), y_test, return_metrics=True)

        acc = metrics['accuracy']
        auc = metrics['auc']

        all_trials.append({
            'model': model_name,
            'auc': auc,
            'accuracy': acc,
            'params': params
        })

        print(f"🎯 Accuracy: {acc:.4f}, AUC: {auc:.4f}")

        if auc > best_score:
            best_score = auc
            best_model = model
            best_params = params

    model_path = f"saved_models/{model_name}_best.pkl"
    if model_name != "MLP":
        joblib.dump(best_model, model_path)
        print(f"💾 Saved best {model_name} model to {model_path}")

    results.append({
        'model': model_name,
        'best_auc': best_score,
        'best_params': best_params,
        'best_model': best_model
    })

print("\n\n====== 🏆 Model Comparison Summary ======")
best_overall = max(results, key=lambda r: r['best_auc'])

for r in results:
    print(f"{r['model']}: AUC = {r['best_auc']:.4f}, Params = {r['best_params']}")

print(f"\n🔥 Best Model: {best_overall['model']} with AUC = {best_overall['best_auc']:.4f}")

model_names = [r['model'] for r in results]
aucs = [r['best_auc'] for r in results]

plt.figure(figsize=(8, 6))
plt.bar(model_names, aucs)
plt.ylabel("Best AUC Score")
plt.title("Best AUC Score per Model")
plt.grid(True)
plt.show()

trial_df = pd.DataFrame(all_trials)

plt.figure(figsize=(10, 6))
for model_name in trial_df['model'].unique():
    subset = trial_df[trial_df['model'] == model_name]
    plt.scatter(subset['accuracy'], subset['auc'], label=model_name, alpha=0.7)

plt.xlabel("Accuracy")
plt.ylabel("AUC Score")
plt.title("All Tuning Trials: Accuracy vs AUC")
plt.legend()
plt.grid(True)
plt.show()

trial_df['param_str'] = trial_df['params'].apply(lambda p: str(p))
fig = px.scatter(trial_df, x='accuracy', y='auc', color='model',
                 hover_data=['param_str'],
                 title='All Tuning Trials with Hover')
fig.show()

trial_df.to_csv("tuning_results.csv", index=False)
print("📁 Tuning results saved to tuning_results.csv")

for model_name in trial_df['model'].unique():
    subset = trial_df[trial_df['model'] == model_name]
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(subset)), subset['auc'])
    plt.title(f"{model_name} - AUC by Param Combination")
    plt.xlabel("Param Combo Index")
    plt.ylabel("AUC Score")
    plt.grid(True)
    plt.show()

if hasattr(best_overall['best_model'], 'feature_importance'):
    best_overall['best_model'].feature_importance(X_train)
