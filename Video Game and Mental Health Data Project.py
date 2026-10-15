"""
CS 432 - Introduction to Applied Machine Learning
Final Project: The Impact of Gaming on Mental Health
Author: Mike Russell (russelm5@oregonstate.edu)
Oregon State University - Spring 2026

This file contains the complete Python code for the entire project, covering:
  - Module 1:    Data gathering and cleaning (run inline at the top of each section)
  - Module 2-3:  Clustering and PCA (unsupervised)
  - Module 4-5:  Linear Regression and Logistic Regression
  - Module 6-7:  Decision Trees and Naive Bayes (Bayesian Classification)
  - Module 8-9:  Support Vector Machines and Ensemble Methods (Random Forest,
                 AdaBoost, Stacking)

Each section can be run independently. All sklearn packages are used as
recommended by the assignment.

Datasets used:
  data1/GamingStudy_data.csv  - Online Gaming Anxiety Study (Sauter/Draschkow 2017)
  data2/online_gaming_behavior_dataset.csv - Predict Online Gaming Behavior (El Kharoua 2024)
"""



# ======================================================================
# MODULE 2 & 3: Clustering and PCA (Unsupervised Learning)
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import os

os.makedirs('images_mod23', exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("CS 432 - Module 2 & 3: Clustering & PCA")
print("=" * 70)

# Load dataset 2
df2 = pd.read_csv('data2/online_gaming_behavior_dataset.csv')
df2 = df2.sample(n=2000, random_state=42).reset_index(drop=True)
df2 = df2.drop(columns=['PlayerID'])

# Use 6 numeric features for clustering (no label)
features = ['Age', 'PlayTimeHours', 'SessionsPerWeek',
            'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
X = df2[features].values

# Min-Max normalization
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X)

# ============================================================
# KMEANS WITHOUT PCA
# ============================================================
print("\n--- KMeans without PCA ---")
inertias = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_norm)
    inertias.append(km.inertia_)

# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(range(2, 8), inertias, 'o-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal k')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images_mod23/elbow.png', dpi=120, bbox_inches='tight')
plt.close()

for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_norm)
    sil = silhouette_score(X_norm, labels)
    print(f"  k={k}: silhouette={sil:.4f}, cluster sizes={np.bincount(labels).tolist()}")

# ============================================================
# KMEANS WITH PCA
# ============================================================
print("\n--- KMeans with PCA (reduce to 3 components) ---")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_norm)
print(f"  Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"  Total explained: {pca.explained_variance_ratio_.sum():.3f}")

for k in [2, 3, 4]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    print(f"  PCA + k={k}: silhouette={sil:.4f}")

# 3D scatter with k=4 PCA clusters
km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_pca4 = km4.fit_predict(X_pca)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['#FF6961', '#77B5FE', '#77DD77', '#B19CD9']
for i in range(4):
    mask = labels_pca4 == i
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
               c=colors[i], label=f'Cluster {i} (n={mask.sum()})', alpha=0.6, s=20)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('KMeans Clustering with PCA (k=4)')
ax.legend()
plt.tight_layout()
plt.savefig('images_mod23/kmeans_pca_3d.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# AGGLOMERATIVE HIERARCHICAL CLUSTERING
# ============================================================
print("\n--- Agglomerative Hierarchical Clustering ---")
# Use a subset of 200 for readable dendrogram
sub_idx = np.random.choice(len(X_norm), size=200, replace=False)
X_sub = X_norm[sub_idx]

linkage_matrix = linkage(X_sub, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90)
plt.axhline(y=3.6, color='red', linestyle='--', label='Cut threshold')
plt.title('Dendrogram - Agglomerative Hierarchical Clustering (Ward Linkage)')
plt.xlabel('Sample Index (or Cluster Size)')
plt.ylabel('Distance (Ward)')
plt.legend()
plt.tight_layout()
plt.savefig('images_mod23/dendrogram.png', dpi=120, bbox_inches='tight')
plt.close()

# Cluster with k=3
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_sub)
sil_agg = silhouette_score(X_sub, labels_agg)
print(f"  Agglomerative k=3: silhouette={sil_agg:.4f}, sizes={np.bincount(labels_agg).tolist()}")

print("\nDone. Module 2/3 plots saved to images_mod23/")


# ======================================================================
# MODULE 4 & 5: Linear Regression and Logistic Regression
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                              accuracy_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay, roc_curve, auc)
import os

os.makedirs('images_mod45', exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("CS 432 - Module 4 & 5: Linear & Logistic Regression")
print("=" * 70)

# ============================================================
# RECREATE CLEANED DATASETS (same logic as Module 1)
# ============================================================
print("\n[Loading and cleaning datasets...]")

# Dataset 1: Online Gaming Anxiety Study
df1 = pd.read_csv('data1/GamingStudy_data.csv', encoding='latin-1', low_memory=False)

drop_cols_d1 = ['S. No.', 'Timestamp', 'earnings', 'whyplay', 'highestleague',
                'Reference', 'accept', 'Birthplace', 'Residence_ISO3',
                'Birthplace_ISO3', 'GAD1', 'GAD2', 'GAD3']
df1_clean = df1.drop(columns=[c for c in drop_cols_d1 if c in df1.columns])
df1_clean = df1_clean.dropna(subset=['GADE'])

numeric_fill = ['Hours', 'streams', 'Narcissism', 'SPIN_T', 'SWL_T', 'GAD_T', 'Age']
for col in numeric_fill:
    if col in df1_clean.columns:
        df1_clean[col] = df1_clean[col].fillna(df1_clean[col].median())

spin_cols = [c for c in df1_clean.columns if c.startswith('SPIN') and c != 'SPIN_T']
for col in spin_cols:
    df1_clean[col] = df1_clean[col].fillna(df1_clean[col].median())

df1_clean = df1_clean.dropna(subset=['Work', 'Degree', 'Game', 'Platform'])
df1_clean = df1_clean[df1_clean['Hours'] <= 80]

def anxiety_label(g):
    if g <= 4: return 'Low'
    if g <= 9: return 'Mild'
    if g <= 14: return 'Moderate'
    return 'Severe'
df1_clean['Anxiety_Level'] = df1_clean['GAD_T'].apply(anxiety_label)

print(f"Dataset 1 cleaned: {df1_clean.shape}")

# Dataset 2: Predict Online Gaming Behavior
df2 = pd.read_csv('data2/online_gaming_behavior_dataset.csv')
df2 = df2.sample(n=2000, random_state=42).reset_index(drop=True)
df2 = df2.drop(columns=['PlayerID'])
print(f"Dataset 2 sampled: {df2.shape}")

# ============================================================
# PART 1: LINEAR REGRESSION
# Predict GAD-7 anxiety score from gaming and demographic features
# ============================================================
print("\n" + "=" * 70)
print("PART 1: LINEAR REGRESSION - Predicting Anxiety (GAD_T) from Dataset 1")
print("=" * 70)

# Feature selection: use behavioral and personality features to predict anxiety
features_lr = ['Hours', 'streams', 'Narcissism', 'Age', 'SPIN_T', 'SWL_T']
target_lr = 'GAD_T'

lr_data = df1_clean[features_lr + [target_lr]].copy()
print(f"\nFeatures: {features_lr}")
print(f"Target: {target_lr} (continuous, range 0-21)")
print(f"Data shape: {lr_data.shape}")

# Save before/after preview images
fig, ax = plt.subplots(figsize=(11, 4))
ax.axis('off')
ax.axis('tight')
preview = lr_data.head(8).round(2)
table = ax.table(cellText=preview.values, colLabels=preview.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)
for i in range(len(preview.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('Linear Regression Input Data (first 8 rows)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('images_mod45/lr_data_preview.png', dpi=120, bbox_inches='tight')
plt.close()

# Train/test split
X_lr = lr_data[features_lr].values
y_lr = lr_data[target_lr].values
X_train, X_test, y_train, y_test = train_test_split(X_lr, y_lr, test_size=0.25, random_state=42)
print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")

# Fit linear regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print("\n--- Linear Regression Equation ---")
print(f"Intercept (b0): {lr_model.intercept_:.4f}")
for name, coef in zip(features_lr, lr_model.coef_):
    print(f"  {name:12s}: {coef:+.4f}")

equation_str = f"GAD_T = {lr_model.intercept_:.3f}"
for name, coef in zip(features_lr, lr_model.coef_):
    sign = "+" if coef >= 0 else "-"
    equation_str += f" {sign} {abs(coef):.3f}*{name}"
print(f"\nFull equation:\n{equation_str}")

# Predictions and metrics
y_pred_train = lr_model.predict(X_train)
y_pred_test = lr_model.predict(X_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"\n--- Linear Regression Metrics ---")
print(f"R-squared (train): {r2_train:.4f}")
print(f"R-squared (test):  {r2_test:.4f}")
print(f"RMSE (test):       {rmse_test:.4f}")
print(f"MAE (test):        {mae_test:.4f}")

# Worked example
print("\n--- Worked Example ---")
example_features = [20.0, 5.0, 3.0, 25.0, 25.0, 20.0]  # Hours, streams, Narcissism, Age, SPIN_T, SWL_T
example_pred = lr_model.intercept_ + sum(c * x for c, x in zip(lr_model.coef_, example_features))
example_pred_check = lr_model.predict([example_features])[0]
print("Example player: Hours=20, streams=5, Narcissism=3, Age=25, SPIN_T=25, SWL_T=20")
print(f"Step-by-step:")
running = lr_model.intercept_
print(f"  Start: {lr_model.intercept_:.3f}")
for name, coef, val in zip(features_lr, lr_model.coef_, example_features):
    contribution = coef * val
    running += contribution
    print(f"  + ({coef:+.4f}) * {val} = {contribution:+.3f}  -> running total: {running:.3f}")
print(f"Predicted GAD_T (anxiety score): {example_pred:.3f}")
print(f"sklearn predict() check:        {example_pred_check:.3f}")

# Visualization 1: Actual vs Predicted
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred_test, alpha=0.3, s=20, edgecolors='none', color='#4472C4')
mn, mx = min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual GAD-7 Score', fontsize=11)
ax.set_ylabel('Predicted GAD-7 Score', fontsize=11)
ax.set_title(f'Linear Regression: Actual vs Predicted Anxiety Score\n(R² = {r2_test:.3f})',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images_mod45/lr_actual_vs_predicted.png', dpi=120, bbox_inches='tight')
plt.close()

# Visualization 2: Residuals
residuals = y_test - y_pred_test
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_pred_test, residuals, alpha=0.3, s=20, edgecolors='none', color='#70AD47')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted GAD-7 Score', fontsize=11)
ax.set_ylabel('Residual (Actual - Predicted)', fontsize=11)
ax.set_title('Linear Regression: Residual Plot', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images_mod45/lr_residuals.png', dpi=120, bbox_inches='tight')
plt.close()

# Visualization 3: Coefficient bar chart
fig, ax = plt.subplots(figsize=(8, 5))
coefs = lr_model.coef_
colors = ['#C00000' if c > 0 else '#0070C0' for c in coefs]
ax.barh(features_lr, coefs, color=colors, edgecolor='black')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient Value', fontsize=11)
ax.set_title('Linear Regression Coefficients\n(Effect on Predicted Anxiety Score)',
             fontsize=12, fontweight='bold')
for i, c in enumerate(coefs):
    ax.text(c + (0.005 if c > 0 else -0.005), i, f'{c:+.3f}',
            va='center', ha='left' if c > 0 else 'right', fontsize=10)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('images_mod45/lr_coefficients.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# PART 2: LOGISTIC REGRESSION
# Predict EngagementLevel (High vs Not-High) from Dataset 2
# ============================================================
print("\n" + "=" * 70)
print("PART 2: LOGISTIC REGRESSION - Predicting High Engagement from Dataset 2")
print("=" * 70)

# Create binary label: High engagement vs Not-High (Low/Medium combined)
df2['HighEngagement'] = (df2['EngagementLevel'] == 'High').astype(int)
print(f"\nLabel distribution:")
print(df2['HighEngagement'].value_counts())
print(f"  0 = Not High (Low or Medium): {(df2['HighEngagement']==0).sum()}")
print(f"  1 = High Engagement:           {(df2['HighEngagement']==1).sum()}")

features_log = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
                'PlayerLevel', 'AchievementsUnlocked', 'InGamePurchases']
target_log = 'HighEngagement'

log_data = df2[features_log + [target_log]].copy()
print(f"\nFeatures: {features_log}")
print(f"Target: {target_log} (binary)")

# Save preview image
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
preview2 = log_data.head(8).round(2)
table = ax.table(cellText=preview2.values, colLabels=preview2.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for i in range(len(preview2.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('Logistic Regression Input Data (first 8 rows)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('images_mod45/log_data_preview.png', dpi=120, bbox_inches='tight')
plt.close()

X_log = log_data[features_log].values
y_log = log_data[target_log].values

# Standardize features so coefficients are comparable
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_log_scaled, y_log, test_size=0.25, random_state=42, stratify=y_log)
print(f"\nTrain set: {X_train2.shape}, Test set: {X_test2.shape}")

log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train2, y_train2)

print("\n--- Logistic Regression Equation (on standardized features) ---")
print(f"Intercept (b0): {log_model.intercept_[0]:.4f}")
for name, coef in zip(features_log, log_model.coef_[0]):
    print(f"  {name:28s}: {coef:+.4f}")

eq_log = f"z = {log_model.intercept_[0]:.3f}"
for name, coef in zip(features_log, log_model.coef_[0]):
    sign = "+" if coef >= 0 else "-"
    eq_log += f" {sign} {abs(coef):.3f}*{name}"
print(f"\nLinear part: {eq_log}")
print("Probability: P(High) = 1 / (1 + e^(-z))   [sigmoid function]")

# Predictions
y_pred2 = log_model.predict(X_test2)
y_proba2 = log_model.predict_proba(X_test2)[:, 1]

acc = accuracy_score(y_test2, y_pred2)
cm = confusion_matrix(y_test2, y_pred2)

print(f"\n--- Logistic Regression Metrics ---")
print(f"Accuracy: {acc:.4f}")
print(f"\nConfusion Matrix:")
print(cm)
print(f"\nClassification Report:")
print(classification_report(y_test2, y_pred2,
                            target_names=['Not High', 'High'], digits=3))

# Worked example
print("\n--- Worked Example ---")
example_raw = [25, 15.0, 18, 90, 75, 40, 1]  # Age, PlayTimeHours, SessionsPerWeek, AvgSession, Level, Achievements, InGamePurchases
example_scaled = scaler.transform([example_raw])[0]
print(f"Example player (raw): Age=25, PlayTimeHours=15, SessionsPerWeek=18,")
print(f"                      AvgSession=90, PlayerLevel=75, Achievements=40, InGamePurchases=1")
print(f"Standardized values:  {[round(v, 3) for v in example_scaled]}")

z = log_model.intercept_[0]
print(f"\nStep-by-step (z = linear combination):")
print(f"  Start (intercept): {log_model.intercept_[0]:.4f}")
for name, coef, val in zip(features_log, log_model.coef_[0], example_scaled):
    contribution = coef * val
    z += contribution
    print(f"  + ({coef:+.4f}) * {val:.3f} = {contribution:+.4f}  -> z: {z:.4f}")

prob = 1 / (1 + np.exp(-z))
print(f"\nApply sigmoid: P(High) = 1 / (1 + e^(-{z:.4f})) = {prob:.4f}")
classification = "High" if prob >= 0.5 else "Not High"
print(f"Threshold 0.5 -> Classification: {classification}")
sklearn_check = log_model.predict_proba([example_scaled])[0, 1]
print(f"sklearn predict_proba check:    {sklearn_check:.4f}")

# Visualization 1: Confusion matrix (Seaborn)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Not High', 'High'], yticklabels=['Not High', 'High'],
            cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Logistic Regression Confusion Matrix\n(Accuracy: {acc:.3f})',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images_mod45/log_confusion_matrix.png', dpi=120, bbox_inches='tight')
plt.close()

# Visualization 2: Sigmoid curve with example
fig, ax = plt.subplots(figsize=(8, 5))
z_range = np.linspace(-6, 6, 200)
sigmoid = 1 / (1 + np.exp(-z_range))
ax.plot(z_range, sigmoid, color='#4472C4', linewidth=2.5, label='Sigmoid: 1/(1+e^(-z))')
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, label='Decision threshold (0.5)')
ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)
ax.scatter([z], [prob], color='red', s=120, zorder=5,
           label=f'Example: z={z:.2f}, P={prob:.3f}', edgecolors='black')
ax.set_xlabel('z (linear combination)', fontsize=11)
ax.set_ylabel('Predicted Probability of High Engagement', fontsize=11)
ax.set_title('Sigmoid Function Mapping z to Probability', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images_mod45/log_sigmoid.png', dpi=120, bbox_inches='tight')
plt.close()

# Visualization 3: ROC curve
fpr, tpr, _ = roc_curve(y_test2, y_proba2)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.plot(fpr, tpr, color='#C00000', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('Logistic Regression ROC Curve', fontsize=12, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('images_mod45/log_roc.png', dpi=120, bbox_inches='tight')
plt.close()

# Visualization 4: Coefficient bar chart
fig, ax = plt.subplots(figsize=(8, 5))
coefs_l = log_model.coef_[0]
colors_l = ['#C00000' if c > 0 else '#0070C0' for c in coefs_l]
ax.barh(features_log, coefs_l, color=colors_l, edgecolor='black')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient Value (standardized)', fontsize=11)
ax.set_title('Logistic Regression Coefficients\n(Effect on Log-Odds of High Engagement)',
             fontsize=12, fontweight='bold')
for i, c in enumerate(coefs_l):
    ax.text(c + (0.02 if c > 0 else -0.02), i, f'{c:+.3f}',
            va='center', ha='left' if c > 0 else 'right', fontsize=10)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('images_mod45/log_coefficients.png', dpi=120, bbox_inches='tight')
plt.close()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Linear Regression  -> R²={r2_test:.3f}, RMSE={rmse_test:.3f}")
print(f"Logistic Regression -> Accuracy={acc:.3f}, AUC={roc_auc:.3f}")
print("\nAll plots saved to: images_mod45/")
print("Done.")


# ======================================================================
# MODULE 6 & 7: Decision Trees and Bayesian Classification
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
import os

os.makedirs('images_mod67', exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("CS 432 - Module 6 & 7: Decision Trees & Naive Bayes")
print("=" * 70)

# ============================================================
# DATA SETUP - same as Module 1
# ============================================================
print("\n[Loading and cleaning Dataset 2: Online Gaming Behavior...]")
df2 = pd.read_csv('data2/online_gaming_behavior_dataset.csv')
df2 = df2.sample(n=2000, random_state=42).reset_index(drop=True)
df2 = df2.drop(columns=['PlayerID'])
print(f"Dataset 2 sampled shape: {df2.shape}")
print(f"EngagementLevel distribution:\n{df2['EngagementLevel'].value_counts()}")

numeric_features = ['Age', 'PlayTimeHours', 'SessionsPerWeek',
                    'AvgSessionDurationMinutes', 'PlayerLevel',
                    'AchievementsUnlocked', 'InGamePurchases']
categorical_features = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
target = 'EngagementLevel'

# ============================================================
# PART 1: DECISION TREE CLASSIFICATION
# Two different trees with different parameters
# ============================================================
print("\n" + "=" * 70)
print("PART 1: DECISION TREE CLASSIFICATION")
print("=" * 70)

# Build the feature matrix: numeric + ordinal-encoded categoricals
df_dt = df2.copy()
cat_encoder = OrdinalEncoder()
df_dt[categorical_features] = cat_encoder.fit_transform(df_dt[categorical_features])

dt_features = numeric_features + categorical_features
X_dt = df_dt[dt_features].values
y_dt = df_dt[target].values

print(f"\nFeatures used: {dt_features}")
print(f"Target: {target} (Low/Medium/High)")
print(f"Total samples: {len(X_dt)}")

X_train, X_test, y_train, y_test = train_test_split(
    X_dt, y_dt, test_size=0.25, random_state=42, stratify=y_dt)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Save preview of input data
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')
preview = df_dt[dt_features + [target]].head(8).round(2)
table = ax.table(cellText=preview.values, colLabels=preview.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)
for i in range(len(preview.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('Decision Tree Input Data (first 8 rows, categoricals ordinal-encoded)',
          fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('images_mod67/dt_data_preview.png', dpi=120, bbox_inches='tight')
plt.close()

# ---------------- TREE 1: max_depth=3, Gini ----------------
print("\n--- Tree 1: max_depth=3, criterion='gini' ---")
tree1 = DecisionTreeClassifier(max_depth=3, criterion='gini', random_state=42)
tree1.fit(X_train, y_train)
y_pred1 = tree1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)
cm1 = confusion_matrix(y_test, y_pred1, labels=['Low', 'Medium', 'High'])
print(f"Accuracy: {acc1:.4f}")
print(f"Tree depth: {tree1.get_depth()}, Leaves: {tree1.get_n_leaves()}")
print(f"\nConfusion matrix (rows=actual, cols=predicted [Low, Medium, High]):")
print(cm1)
print(f"\nClassification report:")
print(classification_report(y_test, y_pred1, digits=3))

# Plot tree 1
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree1, feature_names=dt_features,
          class_names=['High', 'Low', 'Medium'],  # sklearn sorts classes alphabetically
          filled=True, rounded=True, fontsize=9, ax=ax)
plt.title('Decision Tree 1: max_depth=3, criterion="gini"',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images_mod67/dt_tree1_structure.png', dpi=110, bbox_inches='tight')
plt.close()

# Confusion matrix 1
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Tree 1 Confusion Matrix (max_depth=3)\nAccuracy: {acc1:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images_mod67/dt_tree1_cm.png', dpi=120, bbox_inches='tight')
plt.close()

# Feature importance 1
imp1 = sorted(zip(dt_features, tree1.feature_importances_), key=lambda t: t[1], reverse=True)
print("\nTree 1 feature importances:")
for name, val in imp1:
    print(f"  {name:30s}: {val:.4f}")

# ---------------- TREE 2: max_depth=7, Entropy ----------------
print("\n--- Tree 2: max_depth=7, criterion='entropy' ---")
tree2 = DecisionTreeClassifier(max_depth=7, criterion='entropy', random_state=42)
tree2.fit(X_train, y_train)
y_pred2 = tree2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
cm2 = confusion_matrix(y_test, y_pred2, labels=['Low', 'Medium', 'High'])
print(f"Accuracy: {acc2:.4f}")
print(f"Tree depth: {tree2.get_depth()}, Leaves: {tree2.get_n_leaves()}")
print(f"\nConfusion matrix:")
print(cm2)
print(f"\nClassification report:")
print(classification_report(y_test, y_pred2, digits=3))

# Plot tree 2 (will be much bigger)
fig, ax = plt.subplots(figsize=(28, 14))
plot_tree(tree2, feature_names=dt_features,
          class_names=['High', 'Low', 'Medium'],
          filled=True, rounded=True, fontsize=7, ax=ax,
          max_depth=4)  # show first 4 levels for readability
plt.title('Decision Tree 2: max_depth=7, criterion="entropy" (showing top 4 levels)',
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images_mod67/dt_tree2_structure.png', dpi=110, bbox_inches='tight')
plt.close()

# Confusion matrix 2
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm2, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Tree 2 Confusion Matrix (max_depth=7)\nAccuracy: {acc2:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images_mod67/dt_tree2_cm.png', dpi=120, bbox_inches='tight')
plt.close()

imp2 = sorted(zip(dt_features, tree2.feature_importances_), key=lambda t: t[1], reverse=True)
print("\nTree 2 feature importances:")
for name, val in imp2:
    print(f"  {name:30s}: {val:.4f}")

# Combined feature importance comparison
fig, ax = plt.subplots(figsize=(10, 6))
sorted_features = [t[0] for t in imp2]
imp1_dict = dict(imp1)
imp2_dict = dict(imp2)
x = np.arange(len(sorted_features))
width = 0.4
ax.barh(x - width/2, [imp1_dict[f] for f in sorted_features], width,
        color='#4472C4', label='Tree 1 (depth=3)')
ax.barh(x + width/2, [imp2_dict[f] for f in sorted_features], width,
        color='#70AD47', label='Tree 2 (depth=7)')
ax.set_yticks(x)
ax.set_yticklabels(sorted_features)
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Feature Importance: Tree 1 vs Tree 2', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('images_mod67/dt_feature_importance.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# PART 2: NAIVE BAYES CLASSIFICATION
# Method 1: GaussianNB on numeric features
# Method 2: CategoricalNB on categorical features
# ============================================================
print("\n" + "=" * 70)
print("PART 2: NAIVE BAYES CLASSIFICATION")
print("=" * 70)

# ----------- Method 1: GaussianNB on numeric features -----------
print("\n--- Method 1: GaussianNB on numeric features ---")
X_gnb = df2[numeric_features].values
y_gnb = df2[target].values

# Save preview
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')
preview_gnb = df2[numeric_features + [target]].head(8).round(2)
table = ax.table(cellText=preview_gnb.values, colLabels=preview_gnb.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for i in range(len(preview_gnb.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('GaussianNB Input Data (numeric features, first 8 rows)',
          fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('images_mod67/nb_gauss_preview.png', dpi=120, bbox_inches='tight')
plt.close()

X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
    X_gnb, y_gnb, test_size=0.25, random_state=42, stratify=y_gnb)

gnb = GaussianNB()
gnb.fit(X_train_g, y_train_g)
y_pred_g = gnb.predict(X_test_g)
y_proba_g = gnb.predict_proba(X_test_g)
acc_g = accuracy_score(y_test_g, y_pred_g)
cm_g = confusion_matrix(y_test_g, y_pred_g, labels=['Low', 'Medium', 'High'])
print(f"Accuracy: {acc_g:.4f}")
print(f"\nConfusion matrix (rows=actual, cols=predicted [Low, Medium, High]):")
print(cm_g)
print(f"\nClassification report:")
print(classification_report(y_test_g, y_pred_g, digits=3))

print(f"\nClasses (sklearn order): {gnb.classes_}")
print(f"Class priors: {dict(zip(gnb.classes_, gnb.class_prior_))}")

print(f"\nLearned feature means per class:")
print(f"{'Feature':<30s} | " + " | ".join(f"{c:>10s}" for c in gnb.classes_))
for i, f in enumerate(numeric_features):
    means = [gnb.theta_[c_idx][i] for c_idx in range(len(gnb.classes_))]
    print(f"{f:<30s} | " + " | ".join(f"{m:>10.3f}" for m in means))

# Show a few example probability outputs
print("\nExample test predictions with probabilities:")
for i in [0, 5, 10, 50, 100]:
    cls = gnb.classes_
    probs = y_proba_g[i]
    actual = y_test_g[i]
    pred = y_pred_g[i]
    p_str = ", ".join(f"{c}={p:.3f}" for c, p in zip(cls, probs))
    print(f"  Sample {i:3d}: Actual={actual}, Predicted={pred}, P({p_str})")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_g, annot=True, fmt='d', cmap='Purples', ax=ax,
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'GaussianNB Confusion Matrix\nAccuracy: {acc_g:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images_mod67/nb_gauss_cm.png', dpi=120, bbox_inches='tight')
plt.close()

# Probability bar chart for example predictions
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
example_indices = [0, 5, 50, 100]
class_order_idx = [list(gnb.classes_).index(c) for c in ['Low', 'Medium', 'High']]
display_classes = ['Low', 'Medium', 'High']
for ax, idx in zip(axes, example_indices):
    probs = y_proba_g[idx][class_order_idx]
    colors = ['#FF6961', '#FFB347', '#77DD77']
    bars = ax.bar(display_classes, probs, color=colors, edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'Sample {idx}\nActual: {y_test_g[idx]}, Pred: {y_pred_g[idx]}',
                 fontsize=10)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
plt.suptitle('GaussianNB: Example Class Probabilities for Test Samples',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images_mod67/nb_gauss_probas.png', dpi=120, bbox_inches='tight')
plt.close()

# ----------- Method 2: CategoricalNB on categorical features -----------
print("\n--- Method 2: CategoricalNB on categorical + discretized features ---")

# For CategoricalNB, all features must be categorical/discrete
# Use the 4 raw categorical features, plus discretize 3 numeric ones
df_cat = df2.copy()
# Discretize numeric features into bins
df_cat['SessionsPerWeek_bin'] = pd.cut(df_cat['SessionsPerWeek'],
                                       bins=[-0.1, 5, 10, 15, 25],
                                       labels=['VeryLow', 'Low', 'Med', 'High'])
df_cat['AvgSession_bin'] = pd.cut(df_cat['AvgSessionDurationMinutes'],
                                  bins=[-0.1, 45, 90, 135, 200],
                                  labels=['Short', 'Med', 'Long', 'VeryLong'])
df_cat['PlayerLevel_bin'] = pd.cut(df_cat['PlayerLevel'],
                                   bins=[-0.1, 25, 50, 75, 100],
                                   labels=['Beginner', 'Inter', 'Advanced', 'Expert'])

cat_input_features = ['Gender', 'Location', 'GameGenre', 'GameDifficulty',
                      'SessionsPerWeek_bin', 'AvgSession_bin', 'PlayerLevel_bin']

# Save preview of raw categorical/binned data
fig, ax = plt.subplots(figsize=(15, 4))
ax.axis('off')
preview_cat_raw = df_cat[cat_input_features + [target]].head(8)
table = ax.table(cellText=preview_cat_raw.values, colLabels=preview_cat_raw.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for i in range(len(preview_cat_raw.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('CategoricalNB Input Data (raw categories + discretized bins, first 8 rows)',
          fontsize=11, pad=15)
plt.tight_layout()
plt.savefig('images_mod67/nb_cat_preview.png', dpi=120, bbox_inches='tight')
plt.close()

# Encode for the model
enc = OrdinalEncoder()
X_cat = enc.fit_transform(df_cat[cat_input_features].astype(str))
y_cat = df_cat[target].values

# Save encoded preview too
encoded_df = pd.DataFrame(X_cat, columns=cat_input_features).astype(int)
encoded_df['EngagementLevel'] = y_cat
fig, ax = plt.subplots(figsize=(15, 4))
ax.axis('off')
preview_cat_enc = encoded_df.head(8)
table = ax.table(cellText=preview_cat_enc.values, colLabels=preview_cat_enc.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for i in range(len(preview_cat_enc.columns)):
    table[(0, i)].set_facecolor('#70AD47')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('CategoricalNB Input Data (ordinal-encoded integers, first 8 rows)',
          fontsize=11, pad=15)
plt.tight_layout()
plt.savefig('images_mod67/nb_cat_data_encoded.png', dpi=120, bbox_inches='tight')
plt.close()

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cat, y_cat, test_size=0.25, random_state=42, stratify=y_cat)

cnb = CategoricalNB()
cnb.fit(X_train_c, y_train_c)
y_pred_c = cnb.predict(X_test_c)
y_proba_c = cnb.predict_proba(X_test_c)
acc_c = accuracy_score(y_test_c, y_pred_c)
cm_c = confusion_matrix(y_test_c, y_pred_c, labels=['Low', 'Medium', 'High'])
print(f"Accuracy: {acc_c:.4f}")
print(f"\nConfusion matrix:")
print(cm_c)
print(f"\nClassification report:")
print(classification_report(y_test_c, y_pred_c, digits=3))

print(f"\nClasses: {cnb.classes_}")
print(f"Class priors (log): {cnb.class_log_prior_}")
print(f"Class priors (raw): {np.exp(cnb.class_log_prior_)}")

# Show a few example probability outputs
print("\nExample test predictions with probabilities:")
for i in [0, 5, 10, 50, 100]:
    cls = cnb.classes_
    probs = y_proba_c[i]
    actual = y_test_c[i]
    pred = y_pred_c[i]
    p_str = ", ".join(f"{c}={p:.3f}" for c, p in zip(cls, probs))
    print(f"  Sample {i:3d}: Actual={actual}, Predicted={pred}, P({p_str})")

# Confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm_c, annot=True, fmt='d', cmap='Oranges', ax=ax,
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'],
            cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'CategoricalNB Confusion Matrix\nAccuracy: {acc_c:.3f}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('images_mod67/nb_cat_cm.png', dpi=120, bbox_inches='tight')
plt.close()

# Probability bar chart
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
class_order_idx_c = [list(cnb.classes_).index(c) for c in ['Low', 'Medium', 'High']]
for ax, idx in zip(axes, example_indices):
    probs = y_proba_c[idx][class_order_idx_c]
    colors = ['#FF6961', '#FFB347', '#77DD77']
    bars = ax.bar(display_classes, probs, color=colors, edgecolor='black')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title(f'Sample {idx}\nActual: {y_test_c[idx]}, Pred: {y_pred_c[idx]}',
                 fontsize=10)
    for bar, p in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
plt.suptitle('CategoricalNB: Example Class Probabilities for Test Samples',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('images_mod67/nb_cat_probas.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Tree 1 (max_depth=3, gini):    Accuracy = {acc1:.4f}")
print(f"Tree 2 (max_depth=7, entropy): Accuracy = {acc2:.4f}")
print(f"GaussianNB (numeric):          Accuracy = {acc_g:.4f}")
print(f"CategoricalNB (categorical):   Accuracy = {acc_c:.4f}")
print("\nAll plots saved to: images_mod67/")
print("Done.")


# ======================================================================
# MODULE 8 & 9: Support Vector Machines and Ensemble Methods
# ======================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              ConfusionMatrixDisplay)
import os

os.makedirs('images_mod89', exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("CS 432 - Module 8 & 9: SVM & Ensemble Methods")
print("=" * 70)

# ============================================================
# DATA SETUP - same as previous modules
# ============================================================
print("\n[Loading and cleaning Dataset 2: Online Gaming Behavior...]")
df2 = pd.read_csv('data2/online_gaming_behavior_dataset.csv')
df2 = df2.sample(n=2000, random_state=42).reset_index(drop=True)
df2 = df2.drop(columns=['PlayerID'])
print(f"Dataset 2 sampled shape: {df2.shape}")
print(f"EngagementLevel distribution:\n{df2['EngagementLevel'].value_counts()}")

numeric_features = ['Age', 'PlayTimeHours', 'SessionsPerWeek',
                    'AvgSessionDurationMinutes', 'PlayerLevel',
                    'AchievementsUnlocked', 'InGamePurchases']
target = 'EngagementLevel'

X = df2[numeric_features].values
y = df2[target].values

# SVM needs scaled data; ensembles don't but it doesn't hurt for consistency
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Unscaled versions for tree-based ensembles (more interpretable tree splits)
X_train_raw, X_test_raw, _, _ = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

labels_order = ['Low', 'Medium', 'High']

# Helper for pretty confusion matrix
def save_cm(cm, title, filename, cmap='Blues'):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=labels_order, yticklabels=labels_order,
                cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'images_mod89/{filename}', dpi=120, bbox_inches='tight')
    plt.close()

# Save data preview (scaled)
fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')
preview_df = pd.DataFrame(X_train[:8], columns=numeric_features).round(3)
preview_df['EngagementLevel'] = y_train[:8]
table = ax.table(cellText=preview_df.values, colLabels=preview_df.columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.6)
for i in range(len(preview_df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(color='white', weight='bold')
plt.title('Standardized Input Data for SVM (first 8 rows)', fontsize=12, pad=15)
plt.tight_layout()
plt.savefig('images_mod89/svm_data_preview.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# PART 1: SUPPORT VECTOR MACHINE - 3 kernels
# ============================================================
print("\n" + "=" * 70)
print("PART 1: SUPPORT VECTOR MACHINE CLASSIFICATION (3 kernels)")
print("=" * 70)

svm_results = {}
kernels = [
    ('linear', 'linear', 1.0),
    ('rbf', 'rbf', 1.0),
    ('poly', 'poly', 1.0),
]

cm_maps = {'linear': 'Blues', 'rbf': 'Greens', 'poly': 'Purples'}

for name, kernel, C in kernels:
    print(f"\n--- SVM with {kernel} kernel (C={C}) ---")
    svm = SVC(kernel=kernel, C=C, random_state=42,
              degree=3 if kernel == 'poly' else 3,
              gamma='scale')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels_order)
    svm_results[kernel] = {'acc': acc, 'cm': cm, 'n_sv': svm.n_support_}
    print(f"Accuracy: {acc:.4f}")
    print(f"Support vectors per class: {dict(zip(svm.classes_, svm.n_support_))}")
    print(f"Confusion matrix [{labels_order}]:")
    print(cm)
    print(classification_report(y_test, y_pred, digits=3))
    save_cm(cm, f'SVM Confusion Matrix ({kernel} kernel)\nAccuracy: {acc:.3f}, C={C}',
            f'svm_cm_{kernel}.png', cmap=cm_maps[kernel])

# SVM accuracy comparison chart
fig, ax = plt.subplots(figsize=(8, 5))
kernel_names = ['Linear', 'RBF', 'Polynomial']
accs = [svm_results['linear']['acc'], svm_results['rbf']['acc'], svm_results['poly']['acc']]
colors = ['#4472C4', '#70AD47', '#7030A0']
bars = ax.bar(kernel_names, accs, color=colors, edgecolor='black')
ax.set_ylim(0, 1)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_title('SVM Accuracy by Kernel (C=1.0)', fontsize=12, fontweight='bold')
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{a:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images_mod89/svm_accuracy_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# PART 2: ENSEMBLE METHODS
# ============================================================
print("\n" + "=" * 70)
print("PART 2: ENSEMBLE CLASSIFICATION")
print("=" * 70)

# ---------------- Random Forest ----------------
print("\n--- Random Forest (100 trees, max_depth=10) ---")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_raw, y_train)
y_pred_rf = rf.predict(X_test_raw)
acc_rf = accuracy_score(y_test, y_pred_rf)
cm_rf = confusion_matrix(y_test, y_pred_rf, labels=labels_order)
print(f"Accuracy: {acc_rf:.4f}")
print(f"Number of trees: {rf.n_estimators}, Max depth: {rf.max_depth}")
print(f"Confusion matrix:")
print(cm_rf)
print(classification_report(y_test, y_pred_rf, digits=3))
save_cm(cm_rf, f'Random Forest Confusion Matrix\nAccuracy: {acc_rf:.3f} (100 trees)',
        'rf_cm.png', cmap='Blues')

# Feature importance for RF
imp_rf = sorted(zip(numeric_features, rf.feature_importances_),
                key=lambda t: t[1], reverse=True)
print("\nRandom Forest feature importances:")
for name, val in imp_rf:
    print(f"  {name:30s}: {val:.4f}")

fig, ax = plt.subplots(figsize=(9, 5))
names = [t[0] for t in imp_rf]
vals = [t[1] for t in imp_rf]
ax.barh(names[::-1], vals[::-1], color='#4472C4', edgecolor='black')
ax.set_xlabel('Feature Importance', fontsize=11)
ax.set_title('Random Forest Feature Importance', fontsize=12, fontweight='bold')
for i, v in enumerate(vals[::-1]):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('images_mod89/rf_importance.png', dpi=120, bbox_inches='tight')
plt.close()

# Random Forest with only 3 trees to visualize
print("\n--- Random Forest with 3 trees (for visualization) ---")
rf3 = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=42)
rf3.fit(X_train_raw, y_train)
for i in range(3):
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(rf3.estimators_[i], feature_names=numeric_features,
              class_names=rf3.classes_, filled=True, rounded=True,
              fontsize=8, ax=ax, max_depth=3)
    plt.title(f'Random Forest - Tree {i+1} of 3 (max_depth=3)',
              fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(f'images_mod89/rf_tree{i+1}.png', dpi=110, bbox_inches='tight')
    plt.close()

# ---------------- AdaBoost ----------------
print("\n--- AdaBoost (100 estimators) ---")
ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train_raw, y_train)
y_pred_ada = ada.predict(X_test_raw)
acc_ada = accuracy_score(y_test, y_pred_ada)
cm_ada = confusion_matrix(y_test, y_pred_ada, labels=labels_order)
print(f"Accuracy: {acc_ada:.4f}")
print(f"Number of estimators: {ada.n_estimators}")
print(f"Confusion matrix:")
print(cm_ada)
print(classification_report(y_test, y_pred_ada, digits=3))
save_cm(cm_ada, f'AdaBoost Confusion Matrix\nAccuracy: {acc_ada:.3f} (100 estimators)',
        'ada_cm.png', cmap='Greens')

# ---------------- Stacking ----------------
print("\n--- Stacking (RF + SVM + GaussianNB -> Logistic Regression) ---")
# Stacking needs scaled data for the SVM base estimator
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
    ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42)),
    ('gnb', GaussianNB()),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5)
stack.fit(X_train, y_train)
y_pred_stack = stack.predict(X_test)
acc_stack = accuracy_score(y_test, y_pred_stack)
cm_stack = confusion_matrix(y_test, y_pred_stack, labels=labels_order)
print(f"Accuracy: {acc_stack:.4f}")
print(f"Base models: Random Forest, RBF SVM, GaussianNB")
print(f"Meta model: Logistic Regression")
print(f"Confusion matrix:")
print(cm_stack)
print(classification_report(y_test, y_pred_stack, digits=3))
save_cm(cm_stack, f'Stacking Confusion Matrix\nAccuracy: {acc_stack:.3f}',
        'stack_cm.png', cmap='Oranges')

# ---------------- Ensemble comparison chart ----------------
fig, ax = plt.subplots(figsize=(9, 5))
ens_names = ['Random\nForest', 'AdaBoost', 'Stacking']
ens_accs = [acc_rf, acc_ada, acc_stack]
colors = ['#4472C4', '#70AD47', '#ED7D31']
bars = ax.bar(ens_names, ens_accs, color=colors, edgecolor='black')
ax.set_ylim(0, 1)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_title('Ensemble Method Accuracy Comparison', fontsize=12, fontweight='bold')
for bar, a in zip(bars, ens_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{a:.3f}', ha='center', fontsize=11, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images_mod89/ensemble_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

# ---------------- Overall comparison (all models this module) ----------------
fig, ax = plt.subplots(figsize=(11, 5))
all_names = ['SVM\nLinear', 'SVM\nRBF', 'SVM\nPoly', 'Random\nForest', 'AdaBoost', 'Stacking']
all_accs = [svm_results['linear']['acc'], svm_results['rbf']['acc'],
            svm_results['poly']['acc'], acc_rf, acc_ada, acc_stack]
colors = ['#8EAADB', '#8EAADB', '#8EAADB', '#A9D18E', '#A9D18E', '#F4B183']
bars = ax.bar(all_names, all_accs, color=colors, edgecolor='black')
ax.set_ylim(0, 1)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_title('All Module 8/9 Models Compared', fontsize=12, fontweight='bold')
for bar, a in zip(bars, all_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{a:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('images_mod89/all_models_comparison.png', dpi=120, bbox_inches='tight')
plt.close()

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"SVM Linear:      {svm_results['linear']['acc']:.4f}")
print(f"SVM RBF:         {svm_results['rbf']['acc']:.4f}")
print(f"SVM Polynomial:  {svm_results['poly']['acc']:.4f}")
print(f"Random Forest:   {acc_rf:.4f}")
print(f"AdaBoost:        {acc_ada:.4f}")
print(f"Stacking:        {acc_stack:.4f}")
print("\nAll plots saved to: images_mod89/")
print("Done.")
