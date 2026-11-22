# %% [markdown]
# # ML Assignment 2 - Classification on UCI Adult (Census Income)
# Trains Logistic Regression, Decision Tree, kNN, Gaussian Naive Bayes and Random Forest
# on the same data and compares Accuracy, AUC, Precision, Recall, F1 and MCC.
#
# Run (from the repo root):  `python model/train_models.py`
# Optional: `--data path/to/file.csv` to use a local CSV (needs an `income` target column).

# %%
import argparse
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42          # fixed seed for split + all models
TEST_SIZE = 0.20           # 80:20 split
IN_NOTEBOOK = "ipykernel" in sys.modules

# --- folders ---------------------------------------------------------------
try:
    ROOT = Path(__file__).resolve().parent.parent
except NameError:  # running inside a notebook
    cwd = Path.cwd()
    ROOT = cwd.parent if cwd.name == "model" else cwd
MODEL_DIR, RESULTS_DIR, DATA_DIR = ROOT / "model", ROOT / "results", ROOT / "data"
for d in (MODEL_DIR, RESULTS_DIR, DATA_DIR):
    d.mkdir(exist_ok=True)

if IN_NOTEBOOK:
    args = argparse.Namespace(data=None)
else:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="optional local CSV with an 'income' column")
    args = ap.parse_args()


def finish(fig, name):
    """Save a figure to results/ and show it when running in a notebook."""
    fig.savefig(RESULTS_DIR / name, dpi=150, bbox_inches="tight")
    if IN_NOTEBOOK:
        plt.show()
    plt.close(fig)


# %% [markdown]
# ## Step 1 - Dataset: UCI Adult (Census Income)
# * **Source:** UCI Machine Learning Repository (also on Kaggle / OpenML).
# * **Task:** predict whether a person earns **>50K** per year (binary classification).
# * **Why chosen:** 14 features (>=12), ~32.5k rows (>=1,000), genuine missing values
#   (`?` in workclass, occupation, native-country) and a clearly imbalanced target (~76/24),
#   so every requirement of the assignment is naturally exercised.

# %%
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
COLS = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"]
CACHE = DATA_DIR / "adult_raw.csv"


def load_data(path=None):
    if path:
        return pd.read_csv(path, na_values="?", skipinitialspace=True)
    if CACHE.exists():
        return pd.read_csv(CACHE)
    try:
        df = pd.read_csv(UCI_URL, names=COLS, na_values="?", skipinitialspace=True)
    except Exception as e:  # fallback: OpenML mirror of the same dataset
        print("UCI download failed (", e, ") - trying OpenML ...")
        from sklearn.datasets import fetch_openml
        df = fetch_openml("adult", version=2, as_frame=True).frame
        df = df.rename(columns={"class": "income"})
        for c in df.select_dtypes("category"):
            df[c] = df[c].astype(object)
    df.to_csv(CACHE, index=False)
    return df


df = load_data(args.data)
df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)
print("\nMissing values per column:\n", df.isna().sum()[df.isna().sum() > 0])
print("Duplicate rows:", df.duplicated().sum())

# %% [markdown]
# ### Target variable and class distribution

# %%
TARGET = "income"
POSITIVE = ">50K"
dist = df[TARGET].value_counts()
print(pd.DataFrame({"count": dist, "percent": (dist / len(df) * 100).round(2)}))

fig, ax = plt.subplots(figsize=(4.5, 3.5))
sns.countplot(x=TARGET, data=df, ax=ax)
ax.set_title("Target class distribution")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
finish(fig, "class_distribution.png")

# %% [markdown]
# ## Step 2 - Preprocessing
# 1. Remove duplicate rows.
# 2. Stratified 80:20 split with a fixed seed (done **before** fitting any transformer).
# 3. Inside a `ColumnTransformer` (fitted on the training set only):
#    * numeric -> median imputation + `StandardScaler`
#    * categorical -> most-frequent imputation + `OneHotEncoder`
#
# Because the transformer lives in each model's `Pipeline`, no information from the
# test set leaks into imputation, scaling or encoding.

# %%
n_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {n_before - len(df)} duplicate rows -> {len(df)} remain")

y = (df[TARGET] == POSITIVE).astype(int)
X = df.drop(columns=[TARGET])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
print("Train:", X_train.shape, " Test:", X_test.shape)
print("Positive rate  train: %.4f | test: %.4f" % (y_train.mean(), y_test.mean()))

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("Numeric:", num_cols)
print("Categorical:", cat_cols)


def make_preprocessor():
    num = Pipeline([("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])


# Save the raw (un-preprocessed) test split, with target, for the submission / demo.
test_out = X_test.copy()
test_out[TARGET] = np.where(y_test == 1, POSITIVE, "<=50K")
test_out.to_csv(ROOT / "test_data.csv", index=False)
print("Saved test_data.csv", test_out.shape)

# %% [markdown]
# ## Step 3 - Models and evaluation
# Positive class = `>50K`. Precision / Recall / F1 are computed for the positive class;
# AUC uses predicted probabilities.

# %%
MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5,
                                            random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                                       n_jobs=-1),
}
FILES = {"Logistic Regression": "logistic_regression", "Decision Tree": "decision_tree",
         "kNN": "knn", "Naive Bayes": "naive_bayes", "Random Forest (Ensemble)": "random_forest"}

rows, probas, preds = [], {}, {}
for name, clf in MODELS.items():
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
    pipe.fit(X_train, y_train)                       # preprocessing fitted on train only
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    probas[name], preds[name] = y_prob, y_pred
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    })
    joblib.dump(pipe, MODEL_DIR / f"{FILES[name]}.joblib")
    print(f"trained + saved: {name}")

results = pd.DataFrame(rows).set_index("ML Model Name").round(4)
results.to_csv(RESULTS_DIR / "metrics.csv")
md = results.reset_index().to_markdown(index=False)  # needs 'tabulate'
(RESULTS_DIR / "results_table.md").write_text(md)
print("\n", md)

# %% [markdown]
# ## Visualisations

# %%
METRICS = list(results.columns)

# 1) Grouped bar chart - all metrics, all models
fig, ax = plt.subplots(figsize=(11, 5))
results.T.plot(kind="bar", ax=ax, width=0.8)
ax.set_ylim(min(0, results.values.min() - 0.05), 1.05)
ax.set_title("Model comparison - all evaluation metrics (test set)")
ax.set_ylabel("Score"); ax.set_xlabel("Metric")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3)
plt.xticks(rotation=0)
finish(fig, "metrics_comparison.png")

# 2) Heatmap
fig, ax = plt.subplots(figsize=(8, 3.6))
sns.heatmap(results, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax)
ax.set_title("Metrics heatmap"); ax.set_ylabel("")
finish(fig, "metrics_heatmap.png")

# 3) ROC curves
fig, ax = plt.subplots(figsize=(6, 5))
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC={results.loc[name, 'AUC']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curves"); ax.legend(fontsize=8)
finish(fig, "roc_curves.png")

# 4) Confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(20, 3.6))
for ax, (name, yp) in zip(axes, preds.items()):
    sns.heatmap(confusion_matrix(y_test, yp), annot=True, fmt="d", cmap="Blues",
                cbar=False, ax=ax, xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"])
    ax.set_title(name, fontsize=10); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
finish(fig, "confusion_matrices.png")

# %% [markdown]
# ## Draft observations (auto-generated from the numbers - edit before submitting)

# %%
ranks = results.rank(ascending=False, method="min")
mean_rank = ranks.mean(axis=1).sort_values()
lines = []
for name in results.index:
    best = [m for m in METRICS if ranks.loc[name, m] == 1]
    worst = [m for m in METRICS if ranks.loc[name, m] == ranks[m].max()]
    p, r = results.loc[name, "Precision"], results.loc[name, "Recall"]
    bal = "balanced precision/recall" if abs(p - r) < 0.05 else (
        "favours recall over precision (over-predicts >50K)" if r > p
        else "favours precision over recall (misses many >50K cases)")
    lines.append(f"- **{name}**: accuracy {results.loc[name,'Accuracy']:.3f}, AUC "
                 f"{results.loc[name,'AUC']:.3f}, MCC {results.loc[name,'MCC']:.3f}; {bal}. "
                 f"Best on: {', '.join(best) or 'none'}; weakest on: {', '.join(worst) or 'none'}.")
winner = mean_rank.index[0]
lines.append(f"\n**Overall winner (best average rank across 6 metrics): {winner}** "
             f"(MCC {results.loc[winner,'MCC']:.3f}, AUC {results.loc[winner,'AUC']:.3f}).")
obs = "\n".join(lines)
(RESULTS_DIR / "auto_observations.md").write_text(obs)
print(obs)
print("\nDone. Outputs in:", RESULTS_DIR)
