"""
MindSignal - Stage 1: Exploratory Data Analysis
Dataset: mindsignal.csv (500 students, social media & mental health survey)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 11

OUTPUT_DIR = 'eda_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n" + "="*70)
print("      MINDSIGNAL - STAGE 1: EXPLORATORY DATA ANALYSIS")
print("="*70 + "\n")

# ── LOAD ─────────────────────────────────────────────────────────────
df = pd.read_csv('.vscoad2/.vscode/mindsignal.csv')
print(f"✓ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── BASIC INFO ────────────────────────────────────────────────────────
print("─"*70)
print("DATASET OVERVIEW")
print("─"*70)
print(df.dtypes.to_string())
print(f"\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "  None")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nBasic Statistics:")
print(df.describe().round(2).to_string())

# ── TARGET SCORE DISTRIBUTIONS ────────────────────────────────────────
print("\n\n─"*70)
print("TARGET SCORE DISTRIBUTIONS")
print("─"*70)

targets = {
    'mental_health_risk':           'Mental Health Risk',
    'sleep_disruption':             'Sleep Disruption',
    'mood_score':                   'Mood Score',
    'content_influence_score':      'Content Influence Score',
    'misinformation_susceptibility':'Misinformation Susceptibility',
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('MindSignal — Target Score Distributions', fontsize=16, fontweight='bold')
colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0', '#f44336']

for i, (col, label) in enumerate(targets.items()):
    ax = axes[i // 3][i % 3]
    sns.histplot(df[col].dropna(), bins=20, ax=ax, color=colors[i], edgecolor='black', alpha=0.8)
    ax.axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f'Mean={df[col].mean():.1f}')
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('Score (0-100)')
    ax.legend(fontsize=9)
    print(f"  {label}: mean={df[col].mean():.1f}  std={df[col].std():.1f}  "
          f"min={df[col].min():.1f}  max={df[col].max():.1f}")

axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_target_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n✓ Saved: 01_target_distributions.png")

# ── FEATURE DISTRIBUTIONS ─────────────────────────────────────────────
numeric_features = ['age', 'year_of_study', 'sleep_hours', 'phone_hours',
                    'daily_notifications', 'exercise_days', 'study_hours']
numeric_features = [c for c in numeric_features if c in df.columns]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('MindSignal — Feature Distributions', fontsize=16, fontweight='bold')
for i, col in enumerate(numeric_features):
    ax = axes[i // 4][i % 4]
    sns.histplot(df[col].dropna(), bins=15, ax=ax, color='#3f51b5', edgecolor='black', alpha=0.8)
    ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
for j in range(len(numeric_features), 8):
    axes[j // 4][j % 4].axis('off')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_feature_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: 02_feature_distributions.png")

# ── CATEGORICAL FEATURES ──────────────────────────────────────────────
cat_features = ['gender', 'platform', 'social_frequency', 'content_type_1', 'content_type_2']
cat_features = [c for c in cat_features if c in df.columns]

fig, axes = plt.subplots(1, len(cat_features), figsize=(22, 6))
fig.suptitle('MindSignal — Categorical Feature Distributions', fontsize=16, fontweight='bold')
for i, col in enumerate(cat_features):
    counts = df[col].value_counts()
    axes[i].bar(counts.index, counts.values, color='#ff9800', edgecolor='black', alpha=0.85)
    axes[i].set_title(col.replace('_', ' ').title(), fontweight='bold')
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_categorical_distributions.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: 03_categorical_distributions.png")

# ── CORRELATION: FEATURES vs TARGETS ──────────────────────────────────
print("\n\n─"*70)
print("FEATURE-TARGET CORRELATIONS")
print("─"*70)

num_df = df[numeric_features + list(targets.keys())].dropna()
corr   = num_df.corr()
target_corr = corr[list(targets.keys())].drop(index=list(targets.keys()))

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(target_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            ax=ax, linewidths=0.5, cbar_kws={'label': 'Pearson r'})
ax.set_title('Feature vs Target Correlations', fontsize=14, fontweight='bold')
ax.set_xticklabels([t.replace('_', '\n') for t in targets.keys()], rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_feature_target_correlations.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print(target_corr.round(3).to_string())
print(f"\n✓ Saved: 04_feature_target_correlations.png")

# ── PHONE HOURS vs MENTAL HEALTH (scatter) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Key Relationships', fontsize=14, fontweight='bold')

axes[0].scatter(df['phone_hours'], df['mental_health_risk'], alpha=0.4, color='#f44336', s=20)
tmp1 = df[['phone_hours','mental_health_risk']].dropna()
m, b = np.polyfit(tmp1['phone_hours'], tmp1['mental_health_risk'], 1)
x_line = np.linspace(df['phone_hours'].min(), df['phone_hours'].max(), 100)
axes[0].plot(x_line, m*x_line+b, 'k--', linewidth=1.5)
axes[0].set_xlabel('Phone Hours / Day'); axes[0].set_ylabel('Mental Health Risk Score')
axes[0].set_title('Phone Use vs Mental Health Risk', fontweight='bold')

axes[1].scatter(df['sleep_hours'], df['mood_score'], alpha=0.4, color='#2196f3', s=20)
tmp2 = df[['sleep_hours','mood_score']].dropna()
m2, b2 = np.polyfit(tmp2['sleep_hours'], tmp2['mood_score'], 1)
x_line2 = np.linspace(df['sleep_hours'].min(), df['sleep_hours'].max(), 100)
axes[1].plot(x_line2, m2*x_line2+b2, 'k--', linewidth=1.5)
axes[1].set_xlabel('Sleep Hours / Day'); axes[1].set_ylabel('Mood Score')
axes[1].set_title('Sleep Hours vs Mood Score', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_key_relationships.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"✓ Saved: 05_key_relationships.png")

# ── SOCIAL FREQUENCY vs TARGETS ───────────────────────────────────────
if 'social_frequency' in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Social Media Frequency vs Mental Health & Misinformation', fontsize=13, fontweight='bold')
    order = ['Rarely', 'Sometimes', 'Often', 'Always']
    order = [o for o in order if o in df['social_frequency'].unique()]
    sns.boxplot(data=df, x='social_frequency', y='mental_health_risk', order=order,
                ax=axes[0], palette='Reds')
    axes[0].set_title('Social Frequency vs Mental Health Risk', fontweight='bold')
    sns.boxplot(data=df, x='social_frequency', y='misinformation_susceptibility', order=order,
                ax=axes[1], palette='Blues')
    axes[1].set_title('Social Frequency vs Misinformation', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/06_social_frequency_boxplots.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: 06_social_frequency_boxplots.png")

# ── SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("✅ EDA COMPLETE")
print("="*70)
print(f"\n📁 Outputs saved to: {OUTPUT_DIR}/")
print(f"   • 01_target_distributions.png")
print(f"   • 02_feature_distributions.png")
print(f"   • 03_categorical_distributions.png")
print(f"   • 04_feature_target_correlations.png")
print(f"   • 05_key_relationships.png")
print(f"   • 06_social_frequency_boxplots.png")
print(f"\n🚀 Next: run 02_train_test_split.py")
print("="*70 + "\n")
"""
MindSignal - Stage 2: Train/Test Split
- Allowlist-only features (pure survey inputs, no derived columns)
- Full categorical encoding before scaling
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def create_train_test_split(file_path='mindsignal.csv', test_size=0.2, random_state=42):

    print("\n" + "="*70)
    print("      MINDSIGNAL - STAGE 2: TRAIN/TEST SPLIT")
    print("="*70 + "\n")

    # ── LOAD ─────────────────────────────────────────────────────────
    print("STEP 1: Loading Data")
    print("-"*70)
    df = pd.read_csv(file_path)
    print(f"✓ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")

    # ── TARGET LABELS ─────────────────────────────────────────────────
    print("STEP 2: Creating Target Labels")
    print("-"*70)

    df['mental_health_risk_label'] = pd.cut(
        df['mental_health_risk'], bins=[0, 33, 66, 100],
        labels=['Low', 'Medium', 'High'], include_lowest=True
    )
    print(f"1. Mental Health Risk:  {df['mental_health_risk_label'].value_counts().to_dict()}")

    df['sleep_disruption_label'] = np.where(df['sleep_disruption'] > 50, 'Yes', 'No')
    print(f"2. Sleep Disruption:    {pd.Series(df['sleep_disruption_label']).value_counts().to_dict()}")

    df['mood_impact_label'] = pd.cut(
        df['mood_score'], bins=[0, 33, 66, 100],
        labels=['Negative', 'Neutral', 'Positive'], include_lowest=True
    )
    print(f"3. Mood Impact:         {df['mood_impact_label'].value_counts().to_dict()}")

    print(f"4. Content Influence:   mean={df['content_influence_score'].mean():.2f}  std={df['content_influence_score'].std():.2f}")

    df['misinformation_susceptibility_label'] = np.where(
        df['misinformation_susceptibility'] > 50, 'Yes', 'No'
    )
    print(f"5. Misinformation:      {pd.Series(df['misinformation_susceptibility_label']).value_counts().to_dict()}\n")

    # ── ENCODE TARGETS ────────────────────────────────────────────────
    print("STEP 3: Encoding Target Labels")
    print("-"*70)

    label_encoders = {}

    le_mh = LabelEncoder()
    df['mental_health_encoded'] = le_mh.fit_transform(df['mental_health_risk_label'].astype(str))
    label_encoders['mental_health_risk'] = le_mh
    print(f"  Mental Health:  {dict(zip(le_mh.classes_, le_mh.transform(le_mh.classes_)))}")

    le_sd = LabelEncoder()
    df['sleep_disruption_encoded'] = le_sd.fit_transform(df['sleep_disruption_label'])
    label_encoders['sleep_disruption'] = le_sd
    print(f"  Sleep:          {dict(zip(le_sd.classes_, le_sd.transform(le_sd.classes_)))}")

    le_mi = LabelEncoder()
    df['mood_impact_encoded'] = le_mi.fit_transform(df['mood_impact_label'].astype(str))
    label_encoders['mood_impact'] = le_mi
    print(f"  Mood Impact:    {dict(zip(le_mi.classes_, le_mi.transform(le_mi.classes_)))}")

    le_ms = LabelEncoder()
    df['misinformation_encoded'] = le_ms.fit_transform(df['misinformation_susceptibility_label'])
    label_encoders['misinformation'] = le_ms
    print(f"  Misinformation: {dict(zip(le_ms.classes_, le_ms.transform(le_ms.classes_)))}\n")

    # ── FEATURE MATRIX ────────────────────────────────────────────────
    print("STEP 4: Building Feature Matrix (Survey Inputs Only)")
    print("-"*70)

    # ALLOWLIST — only raw survey inputs, nothing derived or computed
    SURVEY_FEATURES = [
        'age',
        'gender',
        'year_of_study',
        'sleep_hours',
        'phone_hours',
        'platform',
        'social_frequency',
        'daily_notifications',
        'exercise_days',
        'study_hours',
        'content_type_1',
        'content_type_2',
    ]

    feature_columns = [c for c in SURVEY_FEATURES if c in df.columns]
    missing_cols = [c for c in SURVEY_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"⚠️  Not found in CSV (skipped): {missing_cols}")

    X = df[feature_columns].copy()

    # Fill missing values
    for col in X.select_dtypes(include='number').columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include='object').columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode()[0])

    # Encode all categorical columns
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[f'feature_{col}'] = le
        print(f"  Encoded: {col} → classes: {list(le.classes_)}")

    # Verify all numeric
    remaining = X.select_dtypes(include='object').columns.tolist()
    if remaining:
        print(f"\n❌ Still non-numeric: {remaining}")
        return None

    feature_columns_final = list(X.columns)
    print(f"\n✅ Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"   Features: {feature_columns_final}")
    ratio = X.shape[0] / X.shape[1]
    print(f"   Row/feature ratio: {ratio:.1f}x  {'✓ Good' if ratio >= 20 else '⚠️  Low'}\n")

    X_np = X.values

    # ── SPLIT ─────────────────────────────────────────────────────────
    print("STEP 5: Train/Test Split (80/20)")
    print("-"*70)

    def safe_split(X, y, stratify=True):
        if stratify:
            try:
                return train_test_split(X, y, test_size=test_size,
                                        random_state=random_state, stratify=y)
            except ValueError:
                print("   ⚠️  Stratify failed — using random split")
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    splits = {}

    y1 = df['mental_health_encoded'].values
    X_tr1, X_te1, y_tr1, y_te1 = safe_split(X_np, y1)
    splits['mental_health'] = {'X_train': X_tr1, 'X_test': X_te1,
                                'y_train': y_tr1, 'y_test': y_te1,
                                'type': 'classification', 'classes': le_mh.classes_}
    print(f"1. Mental Health     — Train: {len(y_tr1)}  Test: {len(y_te1)}  dist: {np.bincount(y_tr1)}")

    y2 = df['sleep_disruption_encoded'].values
    X_tr2, X_te2, y_tr2, y_te2 = safe_split(X_np, y2)
    splits['sleep_disruption'] = {'X_train': X_tr2, 'X_test': X_te2,
                                   'y_train': y_tr2, 'y_test': y_te2,
                                   'type': 'classification', 'classes': le_sd.classes_}
    print(f"2. Sleep Disruption  — Train: {len(y_tr2)}  Test: {len(y_te2)}  dist: {np.bincount(y_tr2)}")

    y3 = df['mood_impact_encoded'].values
    X_tr3, X_te3, y_tr3, y_te3 = safe_split(X_np, y3)
    splits['mood_impact'] = {'X_train': X_tr3, 'X_test': X_te3,
                              'y_train': y_tr3, 'y_test': y_te3,
                              'type': 'classification', 'classes': le_mi.classes_}
    print(f"3. Mood Impact       — Train: {len(y_tr3)}  Test: {len(y_te3)}  dist: {np.bincount(y_tr3)}")

    y4 = df['content_influence_score'].values
    X_tr4, X_te4, y_tr4, y_te4 = safe_split(X_np, y4, stratify=False)
    splits['content_influence'] = {'X_train': X_tr4, 'X_test': X_te4,
                                    'y_train': y_tr4, 'y_test': y_te4,
                                    'type': 'regression'}
    print(f"4. Content Influence — Train: {len(y_tr4)}  Test: {len(y_te4)}  mean={y_tr4.mean():.2f}")

    y5 = df['misinformation_encoded'].values
    X_tr5, X_te5, y_tr5, y_te5 = safe_split(X_np, y5)
    splits['misinformation'] = {'X_train': X_tr5, 'X_test': X_te5,
                                 'y_train': y_tr5, 'y_test': y_te5,
                                 'type': 'classification', 'classes': le_ms.classes_}
    print(f"5. Misinformation    — Train: {len(y_tr5)}  Test: {len(y_te5)}  dist: {np.bincount(y_tr5)}")

    # ── SCALE ─────────────────────────────────────────────────────────
    print(f"\nSTEP 6: Scaling Features (fit on train only)")
    print("-"*70)

    scaler = StandardScaler()
    scaler.fit(splits['mental_health']['X_train'])
    for s in splits.values():
        s['X_train_scaled'] = scaler.transform(s['X_train'])
        s['X_test_scaled']  = scaler.transform(s['X_test'])
    print("✓ StandardScaler applied\n")

    # ── SAVE ──────────────────────────────────────────────────────────
    print("STEP 7: Saving Files")
    print("-"*70)

    with open('mindsignal_train_test_splits.pkl', 'wb') as f:
        pickle.dump(splits, f)
    with open('mindsignal_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('mindsignal_label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('mindsignal_feature_names.pkl', 'wb') as f:
        pickle.dump(feature_columns_final, f)
    df.to_csv('mindsignal_with_labels.csv', index=False)

    print("✓ mindsignal_train_test_splits.pkl")
    print("✓ mindsignal_scaler.pkl")
    print("✓ mindsignal_label_encoders.pkl")
    print("✓ mindsignal_feature_names.pkl")
    print("✓ mindsignal_with_labels.csv")

    # ── VISUALISE ─────────────────────────────────────────────────────
    print(f"\nSTEP 8: Visualizations")
    print("-"*70)

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('MindSignal — Train/Test Split Distributions', fontsize=15, fontweight='bold')
    w = 0.35

    def plot_clf(ax, y_tr, y_te, classes, title, color):
        x  = np.arange(len(classes))
        tc = np.bincount(y_tr, minlength=len(classes))
        ec = np.bincount(y_te, minlength=len(classes))
        ax.bar(x-w/2, tc, w, label='Train', color=color,     alpha=0.8, edgecolor='black')
        ax.bar(x+w/2, ec, w, label='Test',  color='#2196f3', alpha=0.8, edgecolor='black')
        ax.set_xticks(x); ax.set_xticklabels([str(c) for c in classes])
        ax.set_title(title, fontweight='bold'); ax.legend(); ax.grid(axis='y', alpha=0.3)

    plot_clf(axes[0,0], y_tr1, y_te1, le_mh.classes_, 'Mental Health Risk',   '#4caf50')
    plot_clf(axes[0,1], y_tr2, y_te2, le_sd.classes_, 'Sleep Disruption',     '#ff9800')
    plot_clf(axes[0,2], y_tr3, y_te3, le_mi.classes_, 'Mood Impact',          '#9c27b0')

    axes[1,0].hist([y_tr4, y_te4], bins=15, label=['Train','Test'],
                   color=['#4caf50','#2196f3'], alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Content Influence Score', fontweight='bold')
    axes[1,0].legend(); axes[1,0].grid(axis='y', alpha=0.3)

    plot_clf(axes[1,1], y_tr5, y_te5, le_ms.classes_, 'Misinformation', '#f44336')

    ax = axes[1,2]; ax.axis('off')
    tbl = ax.table(
        cellText=[
            ['Mental Health',  len(y_tr1), len(y_te1)],
            ['Sleep',          len(y_tr2), len(y_te2)],
            ['Mood Impact',    len(y_tr3), len(y_te3)],
            ['Content Infl.',  len(y_tr4), len(y_te4)],
            ['Misinformation', len(y_tr5), len(y_te5)],
        ],
        colLabels=['Target', 'Train', 'Test'],
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1.2, 2)
    for j in range(3):
        tbl[(0,j)].set_facecolor('#3f51b5')
        tbl[(0,j)].set_text_props(weight='bold', color='white')
    ax.set_title('Split Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig('mindsignal_train_test_splits.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ mindsignal_train_test_splits.png")

    print("\n" + "="*70)
    print("✅ SPLIT COMPLETE")
    print("="*70)
    print(f"\n📊 Features ({len(feature_columns_final)}): {feature_columns_final}")
    print(f"   Total: {X_np.shape[0]}  Train: {len(y_tr1)}  Test: {len(y_te1)}")
    print(f"\n🚀 Next: run 03_model_training.py")
    print("="*70 + "\n")

    return splits, scaler, label_encoders, feature_columns_final


if __name__ == "__main__":
    import traceback
    try:
        create_train_test_split(file_path='.vscoad2/.vscode/mindsignal.csv', test_size=0.2, random_state=42)
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        traceback.print_exc()
        """
MindSignal - Model Training & Evaluation (Fixed)
Loads from: mindsignal_train_test_splits.pkl
Fixes: overfitting via depth limits, cross-validation, baseline comparisons
"""

import pickle
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    r2_score, mean_absolute_error, mean_squared_error,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# LOAD SPLITS
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("      MINDSIGNAL - MODEL TRAINING & EVALUATION (FIXED)")
print("="*70 + "\n")

print("Loading data...")
try:
    with open('mindsignal_train_test_splits.pkl', 'rb') as f:
        splits = pickle.load(f)
    print("✓ Splits loaded\n")
except FileNotFoundError:
    print("✗ Error: mindsignal_train_test_splits.pkl not found!")
    print("   Run the train/test split script first.\n")
    exit()

# ─────────────────────────────────────────────
# MODEL CANDIDATES
# Depth is deliberately limited to prevent memorization
# ─────────────────────────────────────────────
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    "Random Forest":       RandomForestClassifier(
                               n_estimators=100,
                               max_depth=5,        # was 10 — too deep for small datasets
                               min_samples_leaf=5, # prevents single-sample leaves
                               random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(
                               n_estimators=100,
                               max_depth=3,
                               learning_rate=0.05,
                               random_state=42),
}

REGRESSORS = {
    "Ridge Regression":  Ridge(alpha=10.0),
    "Random Forest":     RandomForestRegressor(
                             n_estimators=100,
                             max_depth=5,
                             min_samples_leaf=5,
                             random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(
                             n_estimators=100,
                             max_depth=3,
                             learning_rate=0.05,
                             random_state=42),
}

# ─────────────────────────────────────────────
# TASK DEFINITIONS
# ─────────────────────────────────────────────
TASKS = {
    "mental_health":    ("classification", "Mental Health Risk"),
    "sleep_disruption": ("classification", "Sleep Disruption"),
    "mood_impact":      ("classification", "Mood Impact"),
    "content_influence":("regression",     "Content Influence Score"),
    "misinformation":   ("classification", "Misinformation Susceptibility"),
}

trained_models = {}
results        = {}

sample_count = len(splits['mental_health']['y_train'])
CV_FOLDS = min(5, max(2, sample_count // 20))
print(f"Using {CV_FOLDS}-fold cross-validation (dataset size: {sample_count} train samples)\n")
print("-" * 70)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
for split_key, (task_type, display_name) in TASKS.items():
    if split_key not in splits:
        print(f"\n⚠️  '{split_key}' not found in splits — skipping.")
        continue

    s    = splits[split_key]
    X_tr = s['X_train_scaled']
    X_te = s['X_test_scaled']
    y_tr = s['y_train']
    y_te = s['y_test']

    print(f"\n{'─'*60}")
    print(f"  {display_name.upper()}  [{task_type}]")
    print(f"{'─'*60}")
    print(f"  Train: {len(y_tr)}  |  Test: {len(y_te)}")

    if task_type == "classification":
        unique, counts = np.unique(y_tr, return_counts=True)
        class_names = s.get('classes', unique)
        print(f"  Classes: {list(class_names)}")
        print(f"  Class distribution (train): { {str(c): int(n) for c, n in zip(class_names, counts)} }")

        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_tr, y_tr)
        baseline_acc = accuracy_score(y_te, dummy.predict(X_te))
        print(f"  Baseline (majority class) accuracy: {baseline_acc:.3f}")

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        target_results = {}

        for name, model in CLASSIFIERS.items():
            try:
                cv_scores = cross_val_score(model, X_tr, y_tr,
                                            cv=cv, scoring="f1_weighted", n_jobs=-1)
                model.fit(X_tr, y_tr)
                y_pred   = model.predict(X_te)
                test_acc = accuracy_score(y_te, y_pred)
                test_f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
                target_results[name] = {
                    "cv_f1_mean": cv_scores.mean(), "cv_f1_std": cv_scores.std(),
                    "test_acc": test_acc, "test_f1": test_f1,
                    "model": model, "y_pred": y_pred,
                }
                flag = "  ⚠️  possible leakage!" if test_acc >= 0.999 else ""
                print(f"  • {name:<22s}  CV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                      f"  |  Test Acc: {test_acc:.3f}  F1: {test_f1:.3f}{flag}")
            except Exception as e:
                print(f"  • {name}: Failed ({str(e)[:50]})")

        if not target_results:
            continue

        best_name = max(target_results, key=lambda n: target_results[n]["cv_f1_mean"])
        best = target_results[best_name]
        trained_models[split_key] = best["model"]

        print(f"\n  ✅  Best: {best_name}  "
              f"(CV F1={best['cv_f1_mean']:.3f}, Test Acc={best['test_acc']:.3f})")
        print(f"\n  Classification Report ({best_name}):")
        print(classification_report(y_te, best["y_pred"],
                                    target_names=[str(c) for c in class_names],
                                    zero_division=0))

        results[split_key] = {
            "task": task_type, "display": display_name, "best_model": best_name,
            "cv_f1": best["cv_f1_mean"], "cv_f1_std": best["cv_f1_std"],
            "test_acc": best["test_acc"], "test_f1": best["test_f1"],
            "baseline": baseline_acc, "classes": class_names,
            "y_test": y_te, "y_pred": best["y_pred"],
        }

    elif task_type == "regression":
        print(f"  Target — mean: {y_tr.mean():.3f}  std: {y_tr.std():.3f}")

        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_tr, y_tr)
        baseline_r2 = r2_score(y_te, dummy.predict(X_te))
        print(f"  Baseline (mean predictor) R²: {baseline_r2:.3f}")

        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
        target_results = {}

        for name, model in REGRESSORS.items():
            try:
                cv_scores = cross_val_score(model, X_tr, y_tr,
                                            cv=cv, scoring="r2", n_jobs=-1)
                model.fit(X_tr, y_tr)
                y_pred    = model.predict(X_te)
                test_r2   = r2_score(y_te, y_pred)
                test_mae  = mean_absolute_error(y_te, y_pred)
                test_rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                target_results[name] = {
                    "cv_r2_mean": cv_scores.mean(), "cv_r2_std": cv_scores.std(),
                    "test_r2": test_r2, "test_mae": test_mae, "test_rmse": test_rmse,
                    "model": model, "y_pred": y_pred,
                }
                flag = "  ⚠️  possible leakage!" if test_r2 >= 0.999 else ""
                print(f"  • {name:<22s}  CV R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
                      f"  |  Test R²: {test_r2:.3f}  MAE: {test_mae:.3f}{flag}")
            except Exception as e:
                print(f"  • {name}: Failed ({str(e)[:50]})")

        if not target_results:
            continue

        best_name = max(target_results, key=lambda n: target_results[n]["cv_r2_mean"])
        best = target_results[best_name]
        trained_models[split_key] = best["model"]

        print(f"\n  ✅  Best: {best_name}  "
              f"(CV R²={best['cv_r2_mean']:.3f}, Test R²={best['test_r2']:.3f}  "
              f"MAE={best['test_mae']:.3f}  RMSE={best['test_rmse']:.3f})")

        results[split_key] = {
            "task": task_type, "display": display_name, "best_model": best_name,
            "cv_r2": best["cv_r2_mean"], "cv_r2_std": best["cv_r2_std"],
            "test_r2": best["test_r2"], "test_mae": best["test_mae"], "test_rmse": best["test_rmse"],
            "baseline_r2": baseline_r2, "y_test": y_te, "y_pred": best["y_pred"],
        }

# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
print(f"\n{'='*70}")
print("  Creating Visualizations...")

sns.set_style("whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('MindSignal Model Performance', fontsize=18, fontweight='bold')

task_list = list(results.keys())
cmaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds']

for i, split_key in enumerate(task_list):
    r = results[split_key]
    row, col = divmod(i, 3)
    ax = axes[row, col]

    if r["task"] == "classification":
        cm = confusion_matrix(r["y_test"], r["y_pred"])
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmaps[i % len(cmaps)], ax=ax,
                    xticklabels=[str(c) for c in r["classes"]],
                    yticklabels=[str(c) for c in r["classes"]])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f"{r['display']}\n{r['best_model']}\n"
                     f"CV F1={r['cv_f1']:.3f}  Test Acc={r['test_acc']:.3f}",
                     fontsize=9, fontweight='bold')
    else:
        ax.scatter(r["y_test"], r["y_pred"], alpha=0.5, color='steelblue', s=20)
        mn = min(r["y_test"].min(), r["y_pred"].min())
        mx = max(r["y_test"].max(), r["y_pred"].max())
        ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Perfect fit')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f"{r['display']}\n{r['best_model']}\n"
                     f"CV R²={r['cv_r2']:.3f}  Test R²={r['test_r2']:.3f}",
                     fontsize=9, fontweight='bold')
        ax.legend(fontsize=7)

# Summary bar chart
ax_sum = axes[1, 2]
names  = [results[k]["display"].replace(" ", "\n") for k in task_list]
scores = [results[k]["cv_f1"] if results[k]["task"] == "classification"
          else max(results[k]["cv_r2"], 0) for k in task_list]
colors = ['#2196f3', '#4caf50', '#ff9800', '#9c27b0', '#f44336']
bars = ax_sum.bar(names, scores, color=colors[:len(names)], edgecolor='black', linewidth=1.2)
for bar, score in zip(bars, scores):
    ax_sum.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax_sum.set_ylim(0, 1.2)
ax_sum.set_ylabel('CV Score (F1 or R²)')
ax_sum.set_title('Model Comparison\n(Cross-Validated)', fontweight='bold')
ax_sum.axhline(0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('mindsignal_performance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ mindsignal_performance.png")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
with open('mindsignal_trained_models.pkl', 'wb') as f:
    pickle.dump(trained_models, f)

save_results = {k: {key: val for key, val in v.items() if key not in ("y_test", "y_pred")}
                for k, v in results.items()}
with open('mindsignal_results.pkl', 'wb') as f:
    pickle.dump(save_results, f)

print("✓ mindsignal_trained_models.pkl")
print("✓ mindsignal_results.pkl")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*70)
print("✅ TRAINING COMPLETE")
print("="*70)
print(f"\n🎯 Trained {len(results)} models:\n")
for k, r in results.items():
    if r["task"] == "classification":
        flag = "  ⚠️  check leakage" if r["test_acc"] >= 0.999 else ""
        print(f"   • {r['display']}: {r['best_model']}"
              f"  (CV F1={r['cv_f1']:.3f}, Test Acc={r['test_acc']:.3f}){flag}")
    else:
        flag = "  ⚠️  check leakage" if r["test_r2"] >= 0.999 else ""
        print(f"   • {r['display']}: {r['best_model']}"
              f"  (CV R²={r['cv_r2']:.3f}, Test R²={r['test_r2']:.3f}){flag}")

print(f"\n📁 Files: mindsignal_trained_models.pkl, mindsignal_results.pkl")
print(f"📊 Visual: mindsignal_performance.png")
print("="*70 + "\n")       