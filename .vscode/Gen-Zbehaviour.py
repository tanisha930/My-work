import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────
df = pd.read_csv('.vscoad2/.vscode/bangladesh_genz_individual_level_dataset.csv')
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# STEP 2 — SELECT RELEVANT COLUMNS
# ─────────────────────────────────────────────────────────────
keep_cols = [
    'source_study',
    # Demographics
    'age', 'gender', 'residence', 'division',
    'monthly_family_income_BDT', 'father_education', 'mother_education',
    # Independent: food chemical exposure
    'fast_food_freq_per_week', 'processed_food_freq',
    'home_cooked_meals_pct', 'childhood_food_exposure_proxy',
    'district_adulteration_level',
    # Independent: air & water pollution
    'district_pm25_level',
    # Independent: parenting style
    'parenting_style',
    # Independent: SES (built from income + education below)
    # Extra context
    'daily_social_media_hours', 'facebook_user',
    'nutrition_literacy', 'food_insecurity',
    'healthy_eating_behavior', 'physical_activity_sufficient',
    # Scale totals
    'phq9_total', 'phq9_category',
    'gad7_total', 'gad7_category',
    'gad7_irritable', 'gad7_restless',
    'perceived_stress_score',
    'ucla_loneliness_total',
    'sleep_duration', 'sleep_quality',
    # Dependent variables
    'impulsivity_proxy',
    'victim_blaming_attitude',
    'antisocial_attitude_score',
    'online_aggression_reported',
]

df = df[keep_cols].copy()
print(f"Selected {len(keep_cols)} columns")

# ─────────────────────────────────────────────────────────────
# STEP 3 — FIX INCONSISTENT INCOME LABELS (from 2 sources)
# ─────────────────────────────────────────────────────────────
income_map = {
    '<10000':      '<10000',
    '10000-20000': '10000-20000',
    '20000-30000': '20000-30000',
    '>30000':      '>30000',
    '<=15000BDT':  '10000-20000',
    '>15000BDT':   '20000-30000',
}
df['monthly_family_income_BDT'] = df['monthly_family_income_BDT'].map(income_map)

# ─────────────────────────────────────────────────────────────
# STEP 4 — IMPUTE MISSING VALUES (mode for categoricals)
# ─────────────────────────────────────────────────────────────
for col in ['food_insecurity', 'father_education', 'mother_education']:
    df[col] = df[col].fillna(df[col].mode()[0])

print(f"Nulls remaining: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 5 — FIX OUTLIERS
# ─────────────────────────────────────────────────────────────
df['perceived_stress_score'] = df['perceived_stress_score'].clip(lower=0, upper=40)

# ─────────────────────────────────────────────────────────────
# STEP 6 — ORDINAL ENCODING
# ─────────────────────────────────────────────────────────────
exposure_map  = {'Low': 0, 'Moderate': 1, 'High': 2, 'Very_High': 3}
proc_map      = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Daily': 4}
nutlit_map    = {'Low': 0, 'Moderate': 1, 'High': 2}
fi_map        = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
sleep_dur_map = {'<4h': 0, '4-6h': 1, '7-8h': 2, '>8h': 3}
sleep_q_map   = {'VeryBad': 0, 'FairlyBad': 1, 'FairlyGood': 2, 'VeryGood': 3}
severity_map  = {'Minimal': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}
edu_map       = {'None': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3}
income_ord    = {'<10000': 0, '10000-20000': 1, '20000-30000': 2, '>30000': 3}

df['childhood_food_exposure_proxy_enc'] = df['childhood_food_exposure_proxy'].map(exposure_map)
df['district_adulteration_level_enc']   = df['district_adulteration_level'].map(exposure_map)
df['district_pm25_level_enc']           = df['district_pm25_level'].map(exposure_map)
df['processed_food_freq_enc']           = df['processed_food_freq'].map(proc_map)
df['nutrition_literacy_enc']            = df['nutrition_literacy'].map(nutlit_map)
df['food_insecurity_enc']               = df['food_insecurity'].map(fi_map)
df['sleep_duration_enc']                = df['sleep_duration'].map(sleep_dur_map)
df['sleep_quality_enc']                 = df['sleep_quality'].map(sleep_q_map)
df['phq9_severity_enc']                 = df['phq9_category'].map(severity_map)
df['gad7_severity_enc']                 = df['gad7_category'].map(severity_map)
df['father_education_enc']              = df['father_education'].map(edu_map)
df['mother_education_enc']              = df['mother_education'].map(edu_map)
df['income_enc']                        = df['monthly_family_income_BDT'].map(income_ord)
df['healthy_eating_enc']                = df['healthy_eating_behavior'].map({'Poor': 0, 'Good': 1})

# ─────────────────────────────────────────────────────────────
# STEP 7 — ONE-HOT ENCODE NOMINAL VARIABLES
# ─────────────────────────────────────────────────────────────
df = pd.get_dummies(df, columns=['gender', 'residence', 'division', 'parenting_style'],
                    drop_first=False, dtype=int)

# ─────────────────────────────────────────────────────────────
# STEP 8 — SES COMPOSITE (income + father edu + mother edu)
# scaled 0–10
# ─────────────────────────────────────────────────────────────
df['ses_composite_scaled'] = (
    (df['income_enc'] + df['father_education_enc'] + df['mother_education_enc']) / 9 * 10
).round(2)

# ─────────────────────────────────────────────────────────────
# STEP 9 — FOOD CHEMICAL EXPOSURE COMPOSITE (0–10)
# ─────────────────────────────────────────────────────────────
df['food_exposure_composite'] = (
    (df['fast_food_freq_per_week'] / 7)               * 0.25 +
    (df['processed_food_freq_enc'] / 4)               * 0.25 +
    (1 - df['home_cooked_meals_pct'] / 100)           * 0.20 +
    (df['district_adulteration_level_enc'] / 3)       * 0.15 +
    (df['childhood_food_exposure_proxy_enc'] / 3)     * 0.15
) * 10
df['food_exposure_composite'] = df['food_exposure_composite'].round(3)

# ─────────────────────────────────────────────────────────────
# STEP 10 — EMOTIONAL DYSREGULATION SCORE / DV2 (0–10)
# PSS 30% + GAD-7 25% + PHQ-9 20% + Loneliness 15% + Sleep 10%
# ─────────────────────────────────────────────────────────────
df['emotional_dysregulation_score'] = (
    (df['perceived_stress_score'] / 40)                * 0.30 +
    (df['gad7_total'] / 21)                            * 0.25 +
    (df['phq9_total'] / 27)                            * 0.20 +
    ((df['ucla_loneliness_total'] - 8) / 24)           * 0.15 +
    (1 - df['sleep_quality_enc'] / 3)                  * 0.10
) * 10
df['emotional_dysregulation_score'] = df['emotional_dysregulation_score'].round(3)

# ─────────────────────────────────────────────────────────────
# STEP 11 — DROP RAW/HELPER COLUMNS
# ─────────────────────────────────────────────────────────────
drop_cols = [
    'processed_food_freq', 'nutrition_literacy', 'food_insecurity',
    'sleep_duration', 'sleep_quality', 'phq9_category', 'gad7_category',
    'father_education', 'mother_education', 'monthly_family_income_BDT',
    'healthy_eating_behavior', 'childhood_food_exposure_proxy',
    'district_adulteration_level', 'district_pm25_level',
]
df.drop(columns=drop_cols, inplace=True)

# ─────────────────────────────────────────────────────────────
# STEP 12 — SAVE
# ─────────────────────────────────────────────────────────────
df.to_csv('bangladesh_genz_CLEANED.csv', index=False)
print(f"Saved: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"Nulls: {df.isnull().sum().sum()}")
print("\nDependent variables:")
for dv in ['impulsivity_proxy', 'emotional_dysregulation_score',
           'antisocial_attitude_score', 'online_aggression_reported']:
    print(f"  {dv}: {df[dv].min():.2f} – {df[dv].max():.2f}")
    """
Statistical Analysis Pipeline
Bangladesh Gen-Z Behaviour Research
======================================
Independent variables : food_exposure_composite, district_pm25_level_enc,
                         parenting_style_*, ses_composite_scaled
Dependent variables   : impulsivity_proxy, emotional_dysregulation_score,
                         antisocial_attitude_score, online_aggression_reported
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import shap

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = 'bangladesh_genz_CLEANED.csv'
OUTPUT_DIR  = '.'          # change to your preferred output folder

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# ── variable groups ─────────────────────────────────────────────────────────────
IVs = [
    'food_exposure_composite',
    'district_pm25_level_enc',
    'ses_composite_scaled',
    'parenting_style_Authoritarian',
    'parenting_style_Authoritative',
    'parenting_style_Neglectful',
    'parenting_style_Permissive',
]

DVs = [
    'impulsivity_proxy',
    'emotional_dysregulation_score',
    'antisocial_attitude_score',
    'online_aggression_reported',
]

SCALE_COLS = [
    'impulsivity_proxy', 'emotional_dysregulation_score',
    'antisocial_attitude_score', 'online_aggression_reported',
    'food_exposure_composite', 'district_pm25_level_enc',
    'ses_composite_scaled', 'perceived_stress_score',
    'phq9_total', 'gad7_total', 'ucla_loneliness_total',
]


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 2 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 2 — DESCRIPTIVE STATISTICS")
print("=" * 70)

desc = df[SCALE_COLS].describe().T
desc['skewness'] = df[SCALE_COLS].skew().round(3)
desc['kurtosis'] = df[SCALE_COLS].kurt().round(3)
desc = desc[['count','mean','std','min','25%','50%','75%','max','skewness','kurtosis']]
desc = desc.round(3)

print(desc.to_string())
desc.to_csv(f'{OUTPUT_DIR}/snippet2_descriptive_stats.csv')
print("\n✓ Saved: snippet2_descriptive_stats.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 3 — CORRELATION MATRIX + HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 3 — CORRELATION MATRIX + HEATMAP")
print("=" * 70)

corr_cols = IVs[:3] + DVs   # IVs (non-dummy) + all DVs
corr_matrix = df[corr_cols].corr(method='pearson').round(3)

print(corr_matrix.to_string())
corr_matrix.to_csv(f'{OUTPUT_DIR}/snippet3_correlation_matrix.csv')

# p-value matrix
n = len(df)
p_matrix = pd.DataFrame(np.ones((len(corr_cols), len(corr_cols))),
                         index=corr_cols, columns=corr_cols)
for i, c1 in enumerate(corr_cols):
    for j, c2 in enumerate(corr_cols):
        if i != j:
            _, p = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
            p_matrix.loc[c1, c2] = round(p, 4)

# heatmap
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.zeros_like(corr_matrix, dtype=bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f',
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    linewidths=0.5, ax=ax,
    annot_kws={'size': 9}
)
ax.set_title('Pearson Correlation Matrix\n(IVs & DVs)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet3_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved: snippet3_correlation_matrix.csv + snippet3_correlation_heatmap.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 4 — INDEPENDENT SAMPLES t-TEST  (Impulsivity by Gender)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 4 — t-TEST: Impulsivity by Gender (Male vs Female)")
print("=" * 70)

def cohens_d(a, b):
    pooled_sd = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_sd

male_imp   = df.loc[df['gender_Male']   == 1, 'impulsivity_proxy'].dropna()
female_imp = df.loc[df['gender_Female'] == 1, 'impulsivity_proxy'].dropna()

t_stat, p_val = stats.ttest_ind(male_imp, female_imp, equal_var=False)  # Welch's t-test
d = cohens_d(male_imp.values, female_imp.values)

print(f"  Male   n={len(male_imp):4d}  mean={male_imp.mean():.3f}  sd={male_imp.std():.3f}")
print(f"  Female n={len(female_imp):4d}  mean={female_imp.mean():.3f}  sd={female_imp.std():.3f}")
print(f"  Welch t = {t_stat:.4f}")
print(f"  p-value  = {p_val:.4f}  {'*** p<0.001' if p_val<0.001 else '** p<0.01' if p_val<0.01 else '* p<0.05' if p_val<0.05 else 'ns'}")
print(f"  Cohen d  = {d:.4f}  ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)")

# run same t-test for all DVs
print("\n  [Extended: t-test for all DVs by gender]")
print(f"  {'DV':<35} {'Male mean':>10} {'Female mean':>12} {'t':>8} {'p':>8} {'d':>8}")
print("  " + "-"*85)
for dv in DVs:
    m = df.loc[df['gender_Male']==1, dv].dropna()
    f = df.loc[df['gender_Female']==1, dv].dropna()
    t, p = stats.ttest_ind(m, f, equal_var=False)
    cd   = cohens_d(m.values, f.values)
    sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {dv:<35} {m.mean():>10.3f} {f.mean():>12.3f} {t:>8.3f} {p:>7.4f}{sig:>3} {cd:>8.4f}")

print("\n✓ t-test complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 5 — ONE-WAY ANOVA + TUKEY POST-HOC
#             DV: antisocial_attitude_score  BY: parenting_style
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 5 — ANOVA: Antisocial Attitude by Parenting Style + Tukey")
print("=" * 70)

# reconstruct parenting style label from dummies
parenting_cols = ['parenting_style_Authoritarian','parenting_style_Authoritative',
                  'parenting_style_Neglectful','parenting_style_Permissive']
df['parenting_label'] = df[parenting_cols].idxmax(axis=1).str.replace('parenting_style_','')

groups = {name: grp['antisocial_attitude_score'].dropna().values
          for name, grp in df.groupby('parenting_label')}

f_stat, p_anova = stats.f_oneway(*groups.values())

print(f"\n  One-way ANOVA — F = {f_stat:.4f},  p = {p_anova:.6f}  "
      f"{'*** p<0.001' if p_anova<0.001 else '** p<0.01' if p_anova<0.01 else '* p<0.05' if p_anova<0.05 else 'ns'}")

print("\n  Group descriptives:")
print(f"  {'Parenting Style':<20} {'n':>6} {'Mean':>8} {'SD':>8}")
print("  " + "-"*45)
for name, vals in groups.items():
    print(f"  {name:<20} {len(vals):>6} {vals.mean():>8.3f} {vals.std():>8.3f}")

# Tukey HSD
print("\n  Tukey HSD Post-Hoc:")
tukey_data   = df[['parenting_label','antisocial_attitude_score']].dropna()
tukey_result = pairwise_tukeyhsd(tukey_data['antisocial_attitude_score'],
                                  tukey_data['parenting_label'], alpha=0.05)
print(tukey_result.summary())

# also run ANOVA for all DVs by parenting
print("\n  [Extended: ANOVA for all DVs by parenting style]")
print(f"  {'DV':<35} {'F':>10} {'p':>12}")
print("  " + "-"*60)
for dv in DVs:
    grps = [grp[dv].dropna().values for _, grp in df.groupby('parenting_label')]
    F, p = stats.f_oneway(*grps)
    sig  = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    print(f"  {dv:<35} {F:>10.4f} {p:>10.6f} {sig}")

print("\n✓ ANOVA complete\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 6 — MULTIPLE LINEAR REGRESSION  (CORE)
#             Run for each DV with all IVs
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 6 — MULTIPLE LINEAR REGRESSION")
print("=" * 70)

reg_IVs = [
    'food_exposure_composite',
    'district_pm25_level_enc',
    'ses_composite_scaled',
    'parenting_style_Authoritarian',
    'parenting_style_Neglectful',
    'parenting_style_Permissive',
    # Authoritative omitted as reference category
    'age', 'gender_Male', 'residence_Urban',
]

regression_results = {}

for dv in ['impulsivity_proxy', 'emotional_dysregulation_score', 'antisocial_attitude_score']:
    print(f"\n  ── DV: {dv} ──")
    formula_vars = ' + '.join(reg_IVs)
    model = smf.ols(f'{dv} ~ {formula_vars}', data=df).fit()

    regression_results[dv] = model

    print(f"  R²        = {model.rsquared:.4f}")
    print(f"  Adj. R²   = {model.rsquared_adj:.4f}")
    print(f"  F-stat    = {model.fvalue:.4f}  p = {model.f_pvalue:.6f}")
    print(f"  N         = {int(model.nobs)}")

    tbl = pd.DataFrame({
        'β (coef)':  model.params.round(4),
        'SE':        model.bse.round(4),
        't':         model.tvalues.round(4),
        'p-value':   model.pvalues.round(4),
        '[0.025':    model.conf_int()[0].round(4),
        '0.975]':    model.conf_int()[1].round(4),
    })
    tbl['sig'] = tbl['p-value'].apply(
        lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')
    print(tbl.to_string())
    tbl.to_csv(f'{OUTPUT_DIR}/snippet6_regression_{dv}.csv')

print("\n✓ Saved: snippet6_regression_*.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 7 — VIF (Multicollinearity Check)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 7 — VIF (Variance Inflation Factor)")
print("=" * 70)

vif_data = df[reg_IVs].dropna()
X_vif = sm.add_constant(vif_data)

vif_df = pd.DataFrame({
    'Variable': X_vif.columns,
    'VIF':      [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_df = vif_df[vif_df['Variable'] != 'const'].sort_values('VIF', ascending=False)
vif_df['VIF'] = vif_df['VIF'].round(3)
vif_df['Status'] = vif_df['VIF'].apply(
    lambda v: '✓ OK' if v < 5 else '⚠ Moderate' if v < 10 else '✗ HIGH — consider dropping')

print(vif_df.to_string(index=False))
vif_df.to_csv(f'{OUTPUT_DIR}/snippet7_vif.csv', index=False)
print("\n  Rule: VIF < 5 = fine | 5–10 = moderate | >10 = multicollinearity problem")
print("✓ Saved: snippet7_vif.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 8 — LOGISTIC REGRESSION + ODDS RATIOS
#             DV: online_aggression_reported (binary)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 8 — LOGISTIC REGRESSION + ODDS RATIOS")
print("=" * 70)

log_formula = 'online_aggression_reported ~ ' + ' + '.join(reg_IVs)
logit_model = smf.logit(log_formula, data=df).fit(disp=0)

or_df = pd.DataFrame({
    'Odds Ratio':    np.exp(logit_model.params).round(4),
    'SE':            logit_model.bse.round(4),
    'z':             logit_model.tvalues.round(4),
    'p-value':       logit_model.pvalues.round(4),
    'OR CI lower':   np.exp(logit_model.conf_int()[0]).round(4),
    'OR CI upper':   np.exp(logit_model.conf_int()[1]).round(4),
})
or_df['sig'] = or_df['p-value'].apply(
    lambda p: '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else '')

print(f"  Pseudo R² (McFadden) = {logit_model.prsquared:.4f}")
print(f"  Log-likelihood       = {logit_model.llf:.4f}")
print(f"  AIC                  = {logit_model.aic:.4f}\n")
print(or_df.to_string())
print("\n  Interpretation: OR > 1 = increases odds of online aggression")
print("                  OR < 1 = decreases odds of online aggression")

or_df.to_csv(f'{OUTPUT_DIR}/snippet8_logistic_odds_ratios.csv')
print("✓ Saved: snippet8_logistic_odds_ratios.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 9 — RANDOM FOREST
#             Target: impulsivity_proxy
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 9 — RANDOM FOREST (Target: impulsivity_proxy)")
print("=" * 70)

rf_features = reg_IVs + [
    'processed_food_freq_enc', 'nutrition_literacy_enc',
    'food_insecurity_enc', 'daily_social_media_hours',
    'healthy_eating_enc', 'sleep_quality_enc',
]

X = df[rf_features].dropna()
y = df.loc[X.index, 'impulsivity_proxy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"  R²   = {r2:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  Train size = {len(X_train)} | Test size = {len(X_test)}\n")

# Feature importance
fi_df = pd.DataFrame({
    'Feature':   rf_features,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False).round(4)

print("  Feature Importance (top 15):")
print(fi_df.head(15).to_string(index=False))
fi_df.to_csv(f'{OUTPUT_DIR}/snippet9_rf_feature_importance.csv', index=False)

# plot
fig, ax = plt.subplots(figsize=(9, 6))
top15 = fi_df.head(15)
bars = ax.barh(top15['Feature'][::-1], top15['Importance'][::-1], color='steelblue', edgecolor='white')
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title(f'Random Forest Feature Importance\nTarget: impulsivity_proxy  |  R²={r2:.3f}', fontsize=12, fontweight='bold')
for bar, val in zip(bars, top15['Importance'][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet9_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: snippet9_rf_feature_importance.csv + .png\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 10 — SHAP VALUES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 10 — SHAP VALUES")
print("=" * 70)

explainer   = shap.TreeExplainer(rf)
shap_sample = X_test.sample(min(300, len(X_test)), random_state=42)
shap_values = explainer.shap_values(shap_sample)

# Summary bar plot
plt.figure()
shap.summary_plot(shap_values, shap_sample, plot_type='bar',
                  feature_names=rf_features, show=False)
plt.title('SHAP Feature Importance (Mean |SHAP|)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet10_shap_bar.png', dpi=150, bbox_inches='tight')
plt.close()

# Beeswarm (dot) plot
plt.figure()
shap.summary_plot(shap_values, shap_sample,
                  feature_names=rf_features, show=False)
plt.title('SHAP Beeswarm — Impact on Impulsivity Prediction', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet10_shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.close()

# Mean SHAP table
shap_df = pd.DataFrame({
    'Feature':        rf_features,
    'Mean |SHAP|':    np.abs(shap_values).mean(axis=0).round(5)
}).sort_values('Mean |SHAP|', ascending=False)
print(shap_df.to_string(index=False))
shap_df.to_csv(f'{OUTPUT_DIR}/snippet10_shap_values.csv', index=False)
print("✓ Saved: snippet10_shap_bar.png + snippet10_shap_beeswarm.png + snippet10_shap_values.csv\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 11 — BOX PLOTS (group comparisons)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 11 — BOX PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dependent Variables by Parenting Style', fontsize=14, fontweight='bold', y=1.01)

palette = {'Authoritarian': '#e74c3c', 'Authoritative': '#2ecc71',
           'Neglectful': '#e67e22', 'Permissive': '#3498db'}

for ax, dv in zip(axes.flatten(), DVs):
    order = ['Authoritative', 'Authoritarian', 'Permissive', 'Neglectful']
    sns.boxplot(data=df, x='parenting_label', y=dv, order=order,
                palette=palette, ax=ax, linewidth=1.2, fliersize=3)
    ax.set_title(dv.replace('_', ' ').title(), fontsize=11, fontweight='bold')
    ax.set_xlabel('Parenting Style', fontsize=9)
    ax.set_ylabel('Score', fontsize=9)
    ax.tick_params(axis='x', labelsize=8, rotation=15)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet11_boxplots_parenting.png', dpi=150, bbox_inches='tight')
plt.close()

# box plots by gender
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Dependent Variables by Gender', fontsize=13, fontweight='bold')

df['gender_label'] = np.where(df['gender_Male']==1, 'Male',
                     np.where(df['gender_Female']==1, 'Female', 'Other'))

for ax, dv in zip(axes, DVs):
    sns.boxplot(data=df, x='gender_label', y=dv,
                palette={'Male':'#3498db','Female':'#e74c3c','Other':'#95a5a6'},
                ax=ax, linewidth=1.2, fliersize=3)
    ax.set_title(dv.replace('_',' ').title(), fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Score', fontsize=9)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet11_boxplots_gender.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: snippet11_boxplots_parenting.png + snippet11_boxplots_gender.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# SNIPPET 12 — SCATTER PLOTS WITH REGRESSION LINE
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SNIPPET 12 — SCATTER PLOTS WITH REGRESSION LINE")
print("=" * 70)

scatter_pairs = [
    ('food_exposure_composite',  'impulsivity_proxy'),
    ('food_exposure_composite',  'antisocial_attitude_score'),
    ('district_pm25_level_enc',  'emotional_dysregulation_score'),
    ('ses_composite_scaled',     'impulsivity_proxy'),
    ('ses_composite_scaled',     'antisocial_attitude_score'),
    ('food_exposure_composite',  'emotional_dysregulation_score'),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('IV–DV Scatter Plots with OLS Regression Line', fontsize=13, fontweight='bold')

for ax, (iv, dv) in zip(axes.flatten(), scatter_pairs):
    x_vals = df[iv]
    y_vals = df[dv]

    # jitter for discrete variables
    jitter_x = x_vals + np.random.uniform(-0.05, 0.05, size=len(x_vals))

    ax.scatter(jitter_x, y_vals, alpha=0.15, s=8, color='steelblue')

    # regression line
    slope, intercept, r, p, se = stats.linregress(x_vals, y_vals)
    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
    ax.plot(x_line, intercept + slope * x_line, color='red', linewidth=2)

    sig = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    ax.set_title(f'{iv}\n→ {dv}\nr={r:.3f}, p{sig}', fontsize=8.5, fontweight='bold')
    ax.set_xlabel(iv.replace('_',' '), fontsize=8)
    ax.set_ylabel(dv.replace('_',' '), fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/snippet12_scatter_regression.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: snippet12_scatter_regression.png\n")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("ALL OUTPUTS SAVED")
print("=" * 70)
outputs = [
    "snippet2_descriptive_stats.csv",
    "snippet3_correlation_matrix.csv",
    "snippet3_correlation_heatmap.png",
    "snippet6_regression_impulsivity_proxy.csv",
    "snippet6_regression_emotional_dysregulation_score.csv",
    "snippet6_regression_antisocial_attitude_score.csv",
    "snippet7_vif.csv",
    "snippet8_logistic_odds_ratios.csv",
    "snippet9_rf_feature_importance.csv",
    "snippet9_rf_feature_importance.png",
    "snippet10_shap_bar.png",
    "snippet10_shap_beeswarm.png",
    "snippet10_shap_values.csv",
    "snippet11_boxplots_parenting.png",
    "snippet11_boxplots_gender.png",
    "snippet12_scatter_regression.png",
]
for f in outputs:
    print(f"  ✓ {f}")
"""
Hypothesis Testing Pipeline
Bangladesh Gen-Z Behaviour Research
=====================================
H1: t-test       — Gender difference in impulsivity
H2: ANOVA        — Parenting style effect on aggression
H3: Regression   — Food safety concern → impulsivity
H4: Regression   — Screen time → online toxicity
H5: Regression   — Fast food → aggression
H6: Logistic     — Combined model predicts high aggression (AUC > 0.70)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('bangladesh_genz_CLEANED.csv')
print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

# reconstruct parenting label from dummies
parenting_cols = ['parenting_style_Authoritarian', 'parenting_style_Authoritative',
                  'parenting_style_Neglectful',    'parenting_style_Permissive']
df['parenting_label'] = (df[parenting_cols]
                         .idxmax(axis=1)
                         .str.replace('parenting_style_', ''))

# reconstruct gender label
df['gender_label'] = np.where(df['gender_Male']==1, 'Male',
                     np.where(df['gender_Female']==1, 'Female', 'Other'))

# binary high-aggression target for H6
median_agg = df['antisocial_attitude_score'].median()
df['high_aggression'] = (df['antisocial_attitude_score'] > median_agg).astype(int)

ALPHA = 0.05

def sep(h):
    print("\n" + "═"*70)
    print(h)
    print("═"*70)

def result_line(reject, p, alpha=ALPHA):
    decision = "REJECT H₀" if reject else "FAIL TO REJECT H₀"
    sig      = '***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'
    return f"  Decision : {decision}  (p={p:.4f} {sig}, α={alpha})"

def cohens_d(a, b):
    pooled = np.sqrt((np.std(a,ddof=1)**2 + np.std(b,ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled else 0


# ══════════════════════════════════════════════════════════════════════════════
# H1 — t-TEST: No gender difference in impulsivity
#      H₀: μ_male = μ_female
#      H₁: μ_male ≠ μ_female  (two-tailed Welch's t-test)
# ══════════════════════════════════════════════════════════════════════════════
sep("H1 — t-TEST: Gender Difference in Impulsivity")

male   = df.loc[df['gender_label']=='Male',   'impulsivity_proxy'].dropna().values
female = df.loc[df['gender_label']=='Female', 'impulsivity_proxy'].dropna().values

t, p_h1 = stats.ttest_ind(male, female, equal_var=False)
d        = cohens_d(male, female)
ci95     = stats.t.interval(0.95, df=min(len(male),len(female))-1,
                             loc=np.mean(male)-np.mean(female),
                             scale=np.sqrt(np.var(male,ddof=1)/len(male) +
                                           np.var(female,ddof=1)/len(female)))

print(f"  H₀ : No gender difference in impulsivity")
print(f"  H₁ : Males ≠ Females in impulsivity (two-tailed)")
print()
print(f"  Male   n={len(male):4d}  mean={np.mean(male):.3f}  SD={np.std(male,ddof=1):.3f}")
print(f"  Female n={len(female):4d}  mean={np.mean(female):.3f}  SD={np.std(female,ddof=1):.3f}")
print(f"  Mean difference       = {np.mean(male)-np.mean(female):.4f}  95% CI [{ci95[0]:.4f}, {ci95[1]:.4f}]")
print(f"  Welch t({len(male)+len(female)-2}) = {t:.4f}")
print(f"  Cohen's d             = {d:.4f}  ({'small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large'} effect)")
print(result_line(p_h1 < ALPHA, p_h1))
print(f"  Expected finding      : Males higher (literature-documented)")
print(f"  Observed              : {'Males higher ✓' if np.mean(male)>np.mean(female) else 'Males NOT higher ✗'}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('H1 — Impulsivity by Gender', fontsize=13, fontweight='bold')

sns.boxplot(data=df[df['gender_label'].isin(['Male','Female'])],
            x='gender_label', y='impulsivity_proxy',
            palette={'Male':'#3498db','Female':'#e74c3c'}, ax=axes[0], linewidth=1.2)
axes[0].set_title(f"Box Plot\nt={t:.3f}, p={p_h1:.4f}, d={d:.3f}", fontsize=10)
axes[0].set_xlabel('Gender'); axes[0].set_ylabel('Impulsivity Score')

for label, vals, color in [('Male', male, '#3498db'), ('Female', female, '#e74c3c')]:
    axes[1].hist(vals, bins=20, alpha=0.55, label=label, color=color, edgecolor='white')
axes[1].axvline(np.mean(male),   color='#3498db', linestyle='--', linewidth=1.8, label=f'Male mean={np.mean(male):.2f}')
axes[1].axvline(np.mean(female), color='#e74c3c', linestyle='--', linewidth=1.8, label=f'Female mean={np.mean(female):.2f}')
axes[1].set_title('Distribution Overlap', fontsize=10)
axes[1].set_xlabel('Impulsivity Score'); axes[1].set_ylabel('Count')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('H1_ttest_gender_impulsivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: H1_ttest_gender_impulsivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# H2 — ANOVA: Parenting style has no effect on aggression
#      H₀: μ_auth = μ_authve = μ_neglect = μ_permis
#      H₁: Authoritarian highest (one-way ANOVA + Tukey)
# ══════════════════════════════════════════════════════════════════════════════
sep("H2 — ANOVA: Parenting Style Effect on Aggression")

groups = {name: grp['antisocial_attitude_score'].dropna().values
          for name, grp in df.groupby('parenting_label')}

F, p_h2 = stats.f_oneway(*groups.values())

# eta-squared effect size
grand_mean = df['antisocial_attitude_score'].mean()
ss_between = sum(len(v)*(np.mean(v)-grand_mean)**2 for v in groups.values())
ss_total   = sum((x-grand_mean)**2 for v in groups.values() for x in v)
eta2       = ss_between / ss_total

print(f"  H₀ : Parenting style has no effect on aggression")
print(f"  H₁ : Authoritarian parenting leads to highest aggression")
print()
print(f"  {'Group':<20} {'n':>5} {'Mean':>8} {'SD':>8}")
print("  " + "-"*45)
for name in ['Authoritarian','Authoritative','Neglectful','Permissive']:
    v = groups[name]
    print(f"  {name:<20} {len(v):>5} {np.mean(v):>8.3f} {np.std(v,ddof=1):>8.3f}")

print(f"\n  One-way ANOVA: F = {F:.4f},  η² = {eta2:.4f}  "
      f"({'small' if eta2<0.06 else 'medium' if eta2<0.14 else 'large'} effect)")
print(result_line(p_h2 < ALPHA, p_h2))

auth_mean  = np.mean(groups['Authoritarian'])
other_mean = np.mean(np.concatenate([groups[k] for k in groups if k!='Authoritarian']))
print(f"  Authoritarian mean={auth_mean:.3f}  vs  others mean={other_mean:.3f}")
print(f"  Expected finding  : Authoritarian highest → "
      f"{'✓ Confirmed' if auth_mean==max(np.mean(v) for v in groups.values()) else '✗ Not confirmed'}")

print("\n  Tukey HSD Post-Hoc:")
tdata  = df[['parenting_label','antisocial_attitude_score']].dropna()
tukey  = pairwise_tukeyhsd(tdata['antisocial_attitude_score'],
                            tdata['parenting_label'], alpha=0.05)
print(tukey.summary())

# plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('H2 — Aggression by Parenting Style', fontsize=13, fontweight='bold')

order   = ['Authoritarian','Authoritative','Neglectful','Permissive']
palette = {'Authoritarian':'#e74c3c','Authoritative':'#2ecc71',
           'Neglectful':'#e67e22','Permissive':'#3498db'}

sns.barplot(data=df, x='parenting_label', y='antisocial_attitude_score',
            order=order, palette=palette, capsize=0.12,
            errorbar='se', ax=axes[0])
axes[0].set_title(f"Mean Aggression ± SE\nF={F:.3f}, p={p_h2:.4f}, η²={eta2:.4f}", fontsize=10)
axes[0].set_xlabel('Parenting Style'); axes[0].set_ylabel('Antisocial Attitude Score')
axes[0].tick_params(axis='x', rotation=15)

sns.violinplot(data=df, x='parenting_label', y='antisocial_attitude_score',
               order=order, palette=palette, inner='box', ax=axes[1])
axes[1].set_title('Distribution by Parenting Style', fontsize=10)
axes[1].set_xlabel('Parenting Style'); axes[1].set_ylabel('Antisocial Attitude Score')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('H2_anova_parenting_aggression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: H2_anova_parenting_aggression.png")


# ══════════════════════════════════════════════════════════════════════════════
# H3 — REGRESSION: Food safety concern → impulsivity
#      H₀: β_food_exposure = 0
#      H₁: β_food_exposure > 0  (one-tailed)
# ══════════════════════════════════════════════════════════════════════════════
sep("H3 — REGRESSION: Food Safety Concern → Impulsivity")

covariates = 'age + gender_Male + residence_Urban + ses_composite_scaled'
formula_h3 = f'impulsivity_proxy ~ food_exposure_composite + district_adulteration_level_enc + {covariates}'
model_h3   = smf.ols(formula_h3, data=df).fit()

b_food = model_h3.params['food_exposure_composite']
p_food = model_h3.pvalues['food_exposure_composite']
p_h3   = p_food / 2  # one-tailed

print(f"  H₀ : Food safety concern has no effect on impulsivity (β = 0)")
print(f"  H₁ : Food safety concern → higher impulsivity (β > 0, one-tailed)")
print()
print(f"  Model: impulsivity ~ food_exposure + district_adulteration + controls")
print(f"  R²         = {model_h3.rsquared:.4f}")
print(f"  Adj. R²    = {model_h3.rsquared_adj:.4f}")
print()
print(f"  β (food_exposure_composite)       = {b_food:.4f}")
print(f"  β (district_adulteration_level)   = {model_h3.params['district_adulteration_level_enc']:.4f}")
print(f"  Two-tailed p = {p_food:.4f}  |  One-tailed p = {p_h3:.4f}")
print(result_line(p_h3 < ALPHA and b_food > 0, p_h3))
print(f"  Expected: positive β  →  Observed β = {b_food:.4f}  "
      f"{'✓ Positive' if b_food>0 else '✗ Negative — unexpected direction'}")

# partial regression plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('H3 — Food Exposure → Impulsivity', fontsize=13, fontweight='bold')

# scatter with OLS line
x = df['food_exposure_composite']
y = df['impulsivity_proxy']
slope, intercept, r, p_r, _ = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 200)
axes[0].scatter(x, y, alpha=0.12, s=10, color='steelblue')
axes[0].plot(x_line, intercept + slope*x_line, color='red', linewidth=2)
axes[0].set_title(f"Scatter + OLS\nr={r:.3f}, β={b_food:.4f}, p={p_food:.4f}", fontsize=10)
axes[0].set_xlabel('Food Chemical Exposure Composite (0–10)')
axes[0].set_ylabel('Impulsivity Score')

# binned means
df['food_bin'] = pd.cut(df['food_exposure_composite'], bins=8)
bin_means = df.groupby('food_bin', observed=True)['impulsivity_proxy'].mean()
axes[1].bar(range(len(bin_means)), bin_means.values, color='steelblue', edgecolor='white')
axes[1].set_xticks(range(len(bin_means)))
axes[1].set_xticklabels([str(b) for b in bin_means.index], rotation=45, fontsize=7)
axes[1].set_title('Mean Impulsivity by Food Exposure Bin', fontsize=10)
axes[1].set_xlabel('Food Exposure Level'); axes[1].set_ylabel('Mean Impulsivity')

plt.tight_layout()
plt.savefig('H3_regression_food_impulsivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: H3_regression_food_impulsivity.png")


# ══════════════════════════════════════════════════════════════════════════════
# H4 — REGRESSION: Screen time → online toxicity (online_aggression_reported)
#      H₀: β_screen_time = 0
#      H₁: β_screen_time > 0  (one-tailed)
# ══════════════════════════════════════════════════════════════════════════════
sep("H4 — REGRESSION: Screen Time → Online Toxicity")

formula_h4 = (f'online_aggression_reported ~ daily_social_media_hours '
              f'+ facebook_user + {covariates}')
model_h4   = smf.ols(formula_h4, data=df).fit()

b_screen = model_h4.params['daily_social_media_hours']
p_screen = model_h4.pvalues['daily_social_media_hours']
p_h4     = p_screen / 2

print(f"  H₀ : Screen time has no effect on online toxicity (β = 0)")
print(f"  H₁ : Screen time → higher online toxicity (β > 0, one-tailed)")
print()
print(f"  Model: online_aggression ~ daily_social_media_hours + controls")
print(f"  R²         = {model_h4.rsquared:.4f}")
print(f"  Adj. R²    = {model_h4.rsquared_adj:.4f}")
print()
print(f"  β (daily_social_media_hours) = {b_screen:.4f}")
print(f"  95% CI [{model_h4.conf_int().loc['daily_social_media_hours',0]:.4f}, "
      f"{model_h4.conf_int().loc['daily_social_media_hours',1]:.4f}]")
print(f"  Two-tailed p = {p_screen:.4f}  |  One-tailed p = {p_h4:.4f}")
print(result_line(p_h4 < ALPHA and b_screen > 0, p_h4))
print(f"  Expected: positive β  →  Observed β = {b_screen:.4f}  "
      f"{'✓ Positive' if b_screen>0 else '✗ Negative — unexpected direction'}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('H4 — Screen Time → Online Toxicity', fontsize=13, fontweight='bold')

x2 = df['daily_social_media_hours']
y2 = df['online_aggression_reported']
slope2, intercept2, r2, p_r2, _ = stats.linregress(x2, y2)
x_line2 = np.linspace(x2.min(), x2.max(), 200)

axes[0].scatter(x2 + np.random.uniform(-0.05,0.05,len(x2)),
                y2 + np.random.uniform(-0.03,0.03,len(y2)),
                alpha=0.08, s=8, color='darkorange')
axes[0].plot(x_line2, intercept2 + slope2*x_line2, color='red', linewidth=2)
axes[0].set_title(f"Scatter + OLS\nr={r2:.3f}, β={b_screen:.4f}, p={p_screen:.4f}", fontsize=10)
axes[0].set_xlabel('Daily Social Media Hours')
axes[0].set_ylabel('Online Aggression (0/1)')

df['screen_bin'] = pd.cut(df['daily_social_media_hours'], bins=6)
sbin = df.groupby('screen_bin', observed=True)['online_aggression_reported'].mean() * 100
axes[1].bar(range(len(sbin)), sbin.values, color='darkorange', edgecolor='white')
axes[1].set_xticks(range(len(sbin)))
axes[1].set_xticklabels([str(b) for b in sbin.index], rotation=45, fontsize=7)
axes[1].set_title('% Reporting Online Aggression by Screen Time', fontsize=10)
axes[1].set_xlabel('Daily Screen Time Bin'); axes[1].set_ylabel('% Reporting Aggression')

plt.tight_layout()
plt.savefig('H4_regression_screentime_toxicity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: H4_regression_screentime_toxicity.png")


# ══════════════════════════════════════════════════════════════════════════════
# H5 — REGRESSION: Fast food → aggression (antisocial_attitude_score)
#      H₀: β_fastfood = 0
#      H₁: β_fastfood > 0  (one-tailed)
# ══════════════════════════════════════════════════════════════════════════════
sep("H5 — REGRESSION: Fast Food Frequency → Aggression")

formula_h5 = (f'antisocial_attitude_score ~ fast_food_freq_per_week '
              f'+ processed_food_freq_enc + food_insecurity_enc '
              f'+ {covariates}')
model_h5   = smf.ols(formula_h5, data=df).fit()

b_ff  = model_h5.params['fast_food_freq_per_week']
p_ff  = model_h5.pvalues['fast_food_freq_per_week']
p_h5  = p_ff / 2

print(f"  H₀ : Fast food has no effect on aggression (β = 0)")
print(f"  H₁ : Fast food → higher aggression (β > 0, one-tailed)")
print()
print(f"  Model: antisocial_attitude ~ fast_food_freq + processed_food + controls")
print(f"  R²         = {model_h5.rsquared:.4f}")
print(f"  Adj. R²    = {model_h5.rsquared_adj:.4f}")
print()
print(f"  β (fast_food_freq_per_week)  = {b_ff:.4f}")
print(f"  β (processed_food_freq_enc)  = {model_h5.params['processed_food_freq_enc']:.4f}")
print(f"  β (food_insecurity_enc)      = {model_h5.params['food_insecurity_enc']:.4f}")
print(f"  95% CI [{model_h5.conf_int().loc['fast_food_freq_per_week',0]:.4f}, "
      f"{model_h5.conf_int().loc['fast_food_freq_per_week',1]:.4f}]")
print(f"  Two-tailed p = {p_ff:.4f}  |  One-tailed p = {p_h5:.4f}")
print(result_line(p_h5 < ALPHA and b_ff > 0, p_h5))
print(f"  Expected: positive β  →  Observed β = {b_ff:.4f}  "
      f"{'✓ Positive' if b_ff>0 else '✗ Negative — unexpected direction'}")

# plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('H5 — Fast Food Frequency → Aggression', fontsize=13, fontweight='bold')

x3 = df['fast_food_freq_per_week']
y3 = df['antisocial_attitude_score']
slope3, intercept3, r3, _, _ = stats.linregress(x3, y3)
x_line3 = np.linspace(x3.min(), x3.max(), 100)

axes[0].scatter(x3 + np.random.uniform(-0.2,0.2,len(x3)),
                y3 + np.random.uniform(-0.2,0.2,len(y3)),
                alpha=0.1, s=8, color='#8e44ad')
axes[0].plot(x_line3, intercept3+slope3*x_line3, color='red', linewidth=2)
axes[0].set_title(f"Scatter + OLS\nr={r3:.3f}, β={b_ff:.4f}, p={p_ff:.4f}", fontsize=10)
axes[0].set_xlabel('Fast Food Days per Week')
axes[0].set_ylabel('Antisocial Attitude Score')

ff_means = df.groupby('fast_food_freq_per_week')['antisocial_attitude_score'].mean()
axes[1].bar(ff_means.index, ff_means.values, color='#8e44ad', edgecolor='white')
axes[1].set_title('Mean Aggression by Fast Food Frequency', fontsize=10)
axes[1].set_xlabel('Fast Food Days per Week')
axes[1].set_ylabel('Mean Antisocial Attitude Score')

plt.tight_layout()
plt.savefig('H5_regression_fastfood_aggression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: H5_regression_fastfood_aggression.png")


# ══════════════════════════════════════════════════════════════════════════════
# H6 — LOGISTIC REGRESSION: Combined model predicts high aggression
#      H₀: AUC = 0.50  (no better than chance)
#      H₁: AUC > 0.70
# ══════════════════════════════════════════════════════════════════════════════
sep("H6 — LOGISTIC REGRESSION: Combined Model Predicts High Aggression (AUC > 0.70)")

features_h6 = [
    'food_exposure_composite',
    'district_pm25_level_enc',
    'district_adulteration_level_enc',
    'ses_composite_scaled',
    'daily_social_media_hours',
    'fast_food_freq_per_week',
    'processed_food_freq_enc',
    'parenting_style_Authoritarian',
    'parenting_style_Neglectful',
    'parenting_style_Permissive',
    'nutrition_literacy_enc',
    'food_insecurity_enc',
    'sleep_quality_enc',
    'healthy_eating_enc',
    'age', 'gender_Male', 'residence_Urban',
]

X_h6 = df[features_h6].dropna()
y_h6 = df.loc[X_h6.index, 'high_aggression']

scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X_h6)

clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

# 10-fold cross-validated AUC
cv     = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
aucs   = cross_val_score(clf, X_scaled, y_h6, cv=cv, scoring='roc_auc')

print(f"  H₀ : Combined model AUC = 0.50 (no better than chance)")
print(f"  H₁ : Combined model AUC > 0.70")
print()
print(f"  Features used  : {len(features_h6)}")
print(f"  Target         : high_aggression (binary, median split)")
print(f"  Positive class : {y_h6.sum()} ({y_h6.mean()*100:.1f}%)")
print(f"  Cross-val      : 10-fold stratified")
print()
print(f"  AUC per fold   : {[round(a,4) for a in aucs]}")
print(f"  Mean AUC       : {aucs.mean():.4f}")
print(f"  SD AUC         : {aucs.std():.4f}")
print(f"  95% CI AUC     : [{aucs.mean()-1.96*aucs.std():.4f}, {aucs.mean()+1.96*aucs.std():.4f}]")

# one-sample t-test: is mean AUC significantly > 0.70?
t_auc, p_auc_twotailed = stats.ttest_1samp(aucs, popmean=0.70)
p_auc_onetailed = p_auc_twotailed / 2

print()
print(f"  One-sample t-test (H₀: μ_AUC = 0.70):")
print(f"    t = {t_auc:.4f},  one-tailed p = {p_auc_onetailed:.4f}")

auc_threshold_met = aucs.mean() > 0.70
print(f"\n  AUC > 0.70 threshold: {'✓ MET' if auc_threshold_met else '✗ NOT MET'}")
print(result_line(auc_threshold_met, aucs.mean()))

# fit final model on full data for ROC curve
clf.fit(X_scaled, y_h6)
y_prob = clf.predict_proba(X_scaled)[:, 1]
y_pred = clf.predict(X_scaled)

fpr, tpr, thresholds = roc_curve(y_h6, y_prob)
final_auc = roc_auc_score(y_h6, y_prob)

print(f"\n  Full-sample AUC  : {final_auc:.4f}")
print(f"\n  Classification Report (threshold=0.5):")
print(classification_report(y_h6, y_pred, target_names=['Low Aggression','High Aggression']))

# coefficient table
clf_coefs = pd.DataFrame({
    'Feature': features_h6,
    'Coefficient': clf.coef_[0].round(4),
    'Odds Ratio': np.exp(clf.coef_[0]).round(4),
}).sort_values('Odds Ratio', ascending=False)
print("  Top predictors (by Odds Ratio):")
print(clf_coefs.to_string(index=False))
clf_coefs.to_csv('H6_logistic_coefficients.csv', index=False)

# plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('H6 — Logistic Regression: Predicting High Aggression', fontsize=13, fontweight='bold')

# ROC curve
axes[0].plot(fpr, tpr, color='#e74c3c', linewidth=2.5, label=f'AUC = {final_auc:.4f}')
axes[0].plot([0,1],[0,1], linestyle='--', color='gray', linewidth=1)
axes[0].axvline(x=0.3, color='steelblue', linestyle=':', linewidth=1, alpha=0.6)
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
axes[0].set_title(f'ROC Curve\nAUC = {final_auc:.4f}  (threshold: 0.70)', fontsize=10)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=10)
axes[0].set_xlim([0,1]); axes[0].set_ylim([0,1.02])

# AUC per fold
fold_nums = [f'F{i+1}' for i in range(len(aucs))]
colors    = ['#2ecc71' if a>0.70 else '#e74c3c' for a in aucs]
axes[1].bar(fold_nums, aucs, color=colors, edgecolor='white')
axes[1].axhline(y=0.70, color='black', linestyle='--', linewidth=1.5, label='AUC = 0.70 threshold')
axes[1].axhline(y=aucs.mean(), color='steelblue', linestyle='-', linewidth=1.5, label=f'Mean = {aucs.mean():.4f}')
axes[1].set_title('Cross-Validated AUC per Fold\n(Green = above 0.70 threshold)', fontsize=10)
axes[1].set_ylabel('AUC'); axes[1].set_ylim([0, 1])
axes[1].legend(fontsize=8)

# confusion matrix
cm = confusion_matrix(y_h6, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low','High'])
disp.plot(ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title('Confusion Matrix\n(Full Sample)', fontsize=10)

plt.tight_layout()
plt.savefig('H6_logistic_combined_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: H6_logistic_combined_model.png + H6_logistic_coefficients.csv")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
sep("HYPOTHESIS TESTING SUMMARY")

summary = [
    ['H1', 't-test', 'Gender diff in impulsivity',
     f't={t:.3f}', f'{p_h1:.4f}', 'REJECT H₀' if p_h1<ALPHA else 'FAIL TO REJECT',
     f"d={d:.3f}", f"{'Males higher ✓' if np.mean(male)>np.mean(female) else 'Not confirmed ✗'}"],
    ['H2', 'ANOVA', 'Parenting → aggression',
     f'F={F:.3f}', f'{p_h2:.4f}', 'REJECT H₀' if p_h2<ALPHA else 'FAIL TO REJECT',
     f"η²={eta2:.4f}", f"{'Authoritarian highest ✓' if np.mean(groups['Authoritarian'])==max(np.mean(v) for v in groups.values()) else 'Not highest ✗'}"],
    ['H3', 'Regression', 'Food exposure → impulsivity',
     f'β={b_food:.4f}', f'{p_h3:.4f}', 'REJECT H₀' if (p_h3<ALPHA and b_food>0) else 'FAIL TO REJECT',
     'One-tailed', f"{'β>0 ✓' if b_food>0 else 'β<0 ✗'}"],
    ['H4', 'Regression', 'Screen time → online toxicity',
     f'β={b_screen:.4f}', f'{p_h4:.4f}', 'REJECT H₀' if (p_h4<ALPHA and b_screen>0) else 'FAIL TO REJECT',
     'One-tailed', f"{'β>0 ✓' if b_screen>0 else 'β<0 ✗'}"],
    ['H5', 'Regression', 'Fast food → aggression',
     f'β={b_ff:.4f}', f'{p_h5:.4f}', 'REJECT H₀' if (p_h5<ALPHA and b_ff>0) else 'FAIL TO REJECT',
     'One-tailed', f"{'β>0 ✓' if b_ff>0 else 'β<0 ✗'}"],
    ['H6', 'Logistic', 'Combined model AUC > 0.70',
     f'AUC={aucs.mean():.4f}', f'{p_auc_onetailed:.4f}', 'REJECT H₀' if auc_threshold_met else 'FAIL TO REJECT',
     f"10-fold CV", f"{'AUC>0.70 ✓' if auc_threshold_met else 'AUC≤0.70 ✗'}"],
]

hdr = ['H', 'Test', 'Description', 'Statistic', 'p-value', 'Decision', 'Effect', 'Expected']
smdf = pd.DataFrame(summary, columns=hdr)
print(smdf.to_string(index=False))
smdf.to_csv('hypothesis_testing_summary.csv', index=False)

print("\n✓ Saved: hypothesis_testing_summary.csv")
print("\nOutput files:")
files = [
    'H1_ttest_gender_impulsivity.png',
    'H2_anova_parenting_aggression.png',
    'H3_regression_food_impulsivity.png',
    'H4_regression_screentime_toxicity.png',
    'H5_regression_fastfood_aggression.png',
    'H6_logistic_combined_model.png',
    'H6_logistic_coefficients.csv',
    'hypothesis_testing_summary.csv',
]
for f in files:
    print(f"  ✓ {f}")   