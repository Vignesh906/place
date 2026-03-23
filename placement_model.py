import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# ── palette ────────────────────────────────────────────────────────────────
PLACED_CLR   = '#00C9A7'
NOTPLACED_CLR = '#FF6B6B'
ACCENT       = '#4E9AF1'
BG           = '#0F1117'
CARD         = '#1A1D27'
TEXT         = '#E8EAF0'
GRID         = '#2A2D3E'

def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor': BG,
        'axes.facecolor':   CARD,
        'axes.edgecolor':   GRID,
        'axes.labelcolor':  TEXT,
        'xtick.color':      TEXT,
        'ytick.color':      TEXT,
        'text.color':       TEXT,
        'grid.color':       GRID,
        'grid.linewidth':   0.6,
        'font.family':      'DejaVu Sans',
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })

# ── load & preprocess ──────────────────────────────────────────────────────
df = pd.read_csv('placement_dataset.csv')
df_encoded = pd.get_dummies(df, columns=['gender','hsc_s','degree_t','workex','specialisation'])
df_encoded['target'] = (df_encoded['status'] == 'Placed').astype(int)
df_encoded.drop('status', axis=1, inplace=True)

features = [c for c in df_encoded.columns if c != 'target']
X = df_encoded[features]
y = df_encoded['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── train models ───────────────────────────────────────────────────────────
models = {
    'Random Forest':     RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN':               KNeighborsClassifier(n_neighbors=7),
}

results = {}
for name, model in models.items():
    Xtr = X_train_sc if name != 'Random Forest' else X_train
    Xte = X_test_sc  if name != 'Random Forest' else X_test
    model.fit(Xtr, y_train)
    preds = model.predict(Xte)
    results[name] = {
        'model':     model,
        'accuracy':  accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds),
        'recall':    recall_score(y_test, preds),
        'f1':        f1_score(y_test, preds),
        'preds':     preds,
    }
    print(f"\n{name}:")
    print(f"  Accuracy : {results[name]['accuracy']:.4f}")
    print(f"  Precision: {results[name]['precision']:.4f}")
    print(f"  Recall   : {results[name]['recall']:.4f}")
    print(f"  F1-Score : {results[name]['f1']:.4f}")

best_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
print(f"\n✅ Best model: {best_name}")

# ── save model & scaler ────────────────────────────────────────────────────
joblib.dump(best_model, 'placement_model.pkl')
joblib.dump(scaler,     'placement_scaler.pkl')
joblib.dump(features,   'placement_features.pkl')
print("Model, scaler, features saved.")

os.makedirs('static/graphs', exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# GRAPH 1 – Model Accuracy Comparison
# ──────────────────────────────────────────────────────────────────────────
set_dark_style()
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(BG)

model_names = list(results.keys())
metrics     = ['accuracy', 'precision', 'recall', 'f1']
metric_lbls = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors      = [PLACED_CLR, ACCENT, '#FFB347', NOTPLACED_CLR]
x = np.arange(len(model_names))
w = 0.18

for i, (m, lbl, clr) in enumerate(zip(metrics, metric_lbls, colors)):
    vals = [results[n][m] for n in model_names]
    bars = ax.bar(x + i*w - 1.5*w, vals, w, label=lbl, color=clr, alpha=0.88,
                  zorder=3, edgecolor='none')
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f'{v:.2f}', ha='center', va='bottom', fontsize=8.5, color=TEXT, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=15, fontweight='bold', pad=16)
ax.legend(loc='upper right', framealpha=0.2, edgecolor=GRID, labelcolor=TEXT)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('static/graphs/model_comparison.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ model_comparison.png saved")

# ──────────────────────────────────────────────────────────────────────────
# GRAPH 2 – Feature Importance (Random Forest)
# ──────────────────────────────────────────────────────────────────────────
set_dark_style()
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)

cmap   = plt.cm.get_cmap('RdYlGn')
colors_bar = [cmap(v / importances.max()) for v in importances.values]
bars   = ax.barh(importances.index, importances.values, color=colors_bar, edgecolor='none', height=0.6, zorder=3)
for bar, v in zip(bars, importances.values):
    ax.text(v + 0.0008, bar.get_y() + bar.get_height()/2,
            f'{v:.4f}', va='center', fontsize=8.5, color=TEXT)

ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=15, fontweight='bold', pad=16)
ax.xaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig('static/graphs/feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ feature_importance.png saved")

# ──────────────────────────────────────────────────────────────────────────
# GRAPH 3 – Placement Distribution
# ──────────────────────────────────────────────────────────────────────────
set_dark_style()
counts = df['status'].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)

# Bar chart
clrs = [PLACED_CLR, NOTPLACED_CLR]
bars = axes[0].bar(counts.index, counts.values, color=clrs, edgecolor='none', width=0.5, zorder=3)
for bar, v in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                 f'{v}\n({v/len(df)*100:.1f}%)', ha='center', va='bottom',
                 fontsize=12, fontweight='bold', color=TEXT)
axes[0].set_title('Placement Distribution', fontsize=14, fontweight='bold', pad=12)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].yaxis.grid(True, zorder=0)
axes[0].set_axisbelow(True)
axes[0].set_ylim(0, counts.max() * 1.2)

# Donut
wedges, texts, autotexts = axes[1].pie(
    counts.values, labels=counts.index, autopct='%1.1f%%',
    colors=clrs, startangle=90, pctdistance=0.78,
    wedgeprops={'width': 0.55, 'edgecolor': BG, 'linewidth': 2.5}
)
for t in texts:   t.set_color(TEXT); t.set_fontsize(12)
for t in autotexts: t.set_color(BG); t.set_fontsize(11); t.set_fontweight('bold')
axes[1].set_title('Placement Split (Donut)', fontsize=14, fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('static/graphs/placement_distribution.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ placement_distribution.png saved")

# ──────────────────────────────────────────────────────────────────────────
# GRAPH 4 – Skills vs Placement (Scatter)
# ──────────────────────────────────────────────────────────────────────────
set_dark_style()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)

placed_df     = df[df['status'] == 'Placed']
not_placed_df = df[df['status'] == 'Not Placed']

for ax, x_col, title in zip(
    axes,
    ['communication', 'technical'],
    ['Communication Skill vs Placement', 'Technical Skill vs Placement']
):
    ax.scatter(not_placed_df[x_col], not_placed_df['coding_score'],
               c=NOTPLACED_CLR, alpha=0.45, s=35, label='Not Placed', edgecolors='none', zorder=3)
    ax.scatter(placed_df[x_col], placed_df['coding_score'],
               c=PLACED_CLR,    alpha=0.55, s=35, label='Placed',     edgecolors='none', zorder=3)
    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=11)
    ax.set_ylabel('Coding Score', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.legend(framealpha=0.2, edgecolor=GRID)
    ax.xaxis.grid(True, zorder=0); ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('static/graphs/skills_vs_placement.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ skills_vs_placement.png saved")

print("\n🎯 All training complete and graphs generated!")
