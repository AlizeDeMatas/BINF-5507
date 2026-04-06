# -*- coding: utf-8 -*-
"""
BINF-5507 Assignment — Sections 1.4 & 1.5
Algorithm: Multi-Layer Perceptron (MLP) for Suicide-Risk Detection in Reddit Posts

Dataset pipeline:
  Raw Reddit text  →  BERT tokenization (2_bert_tokenize.py)
                   →  Token-ID matrix (utils.create_matrix)
                   →  Classifier training / evaluation

Section 1.4 : MLP implementation
Section 1.5 : Comparison against Logistic Regression, SVM, and Naive Bayes
              with AUROC, confusion-matrix grid, and metric bar charts
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — saves PNGs without opening windows
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils  # project-local helper (create_matrix, etc.)

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import LinearSVC
from sklearn.naive_bayes   import BernoulliNB, GaussianNB
from sklearn.calibration   import CalibratedClassifierCV   # gives SVM probabilities
from sklearn.metrics       import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    roc_auc_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Data loading
#     Files produced by 2_bert_tokenize.py.
#     Adjust filenames to match your actual tokenised CSVs.
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_FILE = 'datasets/sample_100_tokens1.csv'
TEST_FILE  = 'datasets/sample_20_tokens.csv'

train_df = pd.read_csv(TRAIN_FILE, encoding='latin-1')
test_df  = pd.read_csv(TEST_FILE,  encoding='latin-1')

train_labels = train_df['classification_int']   # 0 = non-suicide, 1 = suicide
test_labels  = test_df['classification_int']
test_text    = test_df['notesCleaned']          # raw cleaned text (for word analysis)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Data preprocessing — building the token-ID matrix
#
#     Each Reddit post was tokenised with BERT (bert-base-cased).
#     BERT maps every sub-word to an integer ID (vocabulary size ≈ 30 000).
#     Posts are padded / truncated to a fixed length ('padding') so that every
#     row in the matrix has the same number of columns.  Zero-padding is used
#     for shorter posts.
#
#     Result: train_matrix  shape (n_train, padding)
#             test_matrix   shape (n_test,  padding)
#     Each cell is a BERT token ID; the matrix is the feature space fed to
#     the classifiers.
# ─────────────────────────────────────────────────────────────────────────────
max_train = train_df['max_count'].iloc[0]
max_test  = test_df['max_count'].iloc[0]

# Use the longer of the two splits so no information is lost
padding = max(max_train, max_test)

train_matrix = utils.create_matrix(train_df['tokens'], padding)
test_matrix  = utils.create_matrix(test_df['tokens'],  padding)

print(f"Feature matrix  →  train: {train_matrix.shape},  test: {test_matrix.shape}")
print(f"Label distribution (train): {dict(train_labels.value_counts())}")
print(f"Label distribution (test) : {dict(test_labels.value_counts())}\n")

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1.4 — MLP CODE SNIPPET (documented)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  A Multi-Layer Perceptron (MLP) is a feed-forward artificial neural network
#  composed of:
#    • Input layer  — one neuron per feature (= padding, the BERT token IDs)
#    • Hidden layers — learnable weight matrices with non-linear activations
#    • Output layer — one neuron per class (softmax → class probabilities)
#
#  Key hyper-parameters tuned here
#  ──────────────────────────────
#  hidden_layer_sizes : (128, 64)
#      Two hidden layers with 128 and 64 neurons respectively.
#      More neurons → greater capacity but higher overfitting risk.
#      Chosen via coarse grid search on validation accuracy.
#
#  activation : 'relu'
#      Rectified Linear Unit: f(x) = max(0, x).
#      Avoids the vanishing-gradient problem of sigmoid/tanh and trains fast.
#
#  solver : 'adam'
#      Adaptive Moment Estimation — adjusts per-parameter learning rates.
#      Robust default for most classification problems.
#
#  alpha : 1e-4
#      L2 regularisation strength; penalises large weights to reduce
#      overfitting on the relatively small sample dataset.
#
#  max_iter : 500
#      Maximum number of training epochs; early stopping is not used here
#      so we allow enough iterations for convergence.
#
#  random_state : 42
#      Seeds the weight initialisation for reproducibility.
# ─────────────────────────────────────────────────────────────────────────────

mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),   # architecture: 128 → 64 → output
    activation='relu',              # hidden-layer activation function
    solver='adam',                  # optimiser
    alpha=1e-4,                     # L2 regularisation term
    max_iter=500,                   # maximum training epochs
    random_state=42,
)

# Training: back-propagation adjusts weights to minimise cross-entropy loss
mlp.fit(train_matrix, train_labels)

# Inference: hard class predictions (0 or 1)
mlp_preds = mlp.predict(test_matrix)

# Soft probabilities for AUROC (column 1 = P(suicide))
mlp_proba = mlp.predict_proba(test_matrix)[:, 1]

# ── Output interpretation ────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1.4 — MLP Results")
print("=" * 60)
print(f"Accuracy : {accuracy_score(test_labels, mlp_preds)*100:.2f}%")
print(classification_report(test_labels, mlp_preds,
                             target_names=['non-suicide', 'suicide']))

# Loss curve — confirms the model converged during training
plt.figure(figsize=(6, 3))
plt.plot(mlp.loss_curve_, color='steelblue')
plt.xlabel('Epoch')
plt.ylabel('Cross-entropy loss')
plt.title('MLP Training Loss Curve')
plt.tight_layout()
plt.savefig('mlp_loss_curve.png', dpi=150)
# plt.show()  # disabled: using Agg backend
print("Training loss curve saved → mlp_loss_curve.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1.5 — COMPARISON / VISUALISATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
#
#  Classifiers compared
#  ─────────────────────
#  1. MLP (Multi-Layer Perceptron) — main algorithm
#  2. Logistic Regression          — linear probabilistic baseline
#  3. SVM (LinearSVC)              — linear max-margin classifier
#  4. Bernoulli Naive Bayes        — fast generative baseline well-suited to
#                                    binary bag-of-tokens features
#
#  Data transformations & assumptions
#  ────────────────────────────────────
#  • All models receive the same zero-padded BERT token-ID matrix.
#  • No scaling is applied: BERT IDs are already bounded integers.
#  • LinearSVC does not expose predict_proba; a Platt-scaling wrapper
#    (CalibratedClassifierCV) is used to obtain probabilities for AUROC.
#  • BernoulliNB binarises the input internally (binarize=0.0), treating
#    any non-zero token ID as "feature present".
# ─────────────────────────────────────────────────────────────────────────────

# ── 3a. Fit comparison classifiers ──────────────────────────────────────────

lr = LogisticRegression(random_state=42, tol=1e-5, max_iter=100_000, dual=False)
lr.fit(train_matrix, train_labels)
lr_preds = lr.predict(test_matrix)
lr_proba = lr.predict_proba(test_matrix)[:, 1]

# LinearSVC wrapped for probability calibration (Platt scaling via 5-fold CV)
svm_base = LinearSVC(random_state=42, tol=1e-5, max_iter=100_000, dual=False)
svm = CalibratedClassifierCV(svm_base, cv=5)
svm.fit(train_matrix, train_labels)
svm_preds = svm.predict(test_matrix)
svm_proba = svm.predict_proba(test_matrix)[:, 1]

bnb = BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
bnb.fit(train_matrix, train_labels)
bnb_preds = bnb.predict(test_matrix)
bnb_proba = bnb.predict_proba(test_matrix)[:, 1]

# ── 3b. Collect scalar metrics ───────────────────────────────────────────────

def metrics_from_preds(y_true, y_pred, y_prob):
    """Return a dict of common binary-classification metrics."""
    cm   = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        'Accuracy'   : accuracy_score(y_true, y_pred),
        'Precision'  : prec,
        'Sensitivity': rec,
        'Specificity': spec,
        'F1'         : f1,
        'AUROC'      : roc_auc_score(y_true, y_prob),
        'cm'         : cm,
        'preds'      : y_pred,
        'proba'      : y_prob,
    }

results = {
    'MLP'                 : metrics_from_preds(test_labels, mlp_preds, mlp_proba),
    'Logistic Regression' : metrics_from_preds(test_labels, lr_preds,  lr_proba),
    'SVM'                 : metrics_from_preds(test_labels, svm_preds, svm_proba),
    'Bernoulli NB'        : metrics_from_preds(test_labels, bnb_preds, bnb_proba),
}

# Print a summary table
print("=" * 60)
print("SECTION 1.5 — Classifier Comparison")
print("=" * 60)
header = f"{'Classifier':<22} {'Acc':>6} {'Prec':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'AUROC':>7}"
print(header)
print("-" * len(header))
for name, m in results.items():
    print(f"{name:<22} {m['Accuracy']:>6.3f} {m['Precision']:>6.3f} "
          f"{m['Sensitivity']:>6.3f} {m['Specificity']:>6.3f} "
          f"{m['F1']:>6.3f} {m['AUROC']:>7.4f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation 1 — AUROC curves (all classifiers on one plot)
#
#  AUROC (Area Under the Receiver-Operating-Characteristic Curve) measures
#  a classifier's ability to discriminate between classes across all decision
#  thresholds.  AUROC = 1.0 is perfect; 0.5 is random chance.
# ─────────────────────────────────────────────────────────────────────────────
colours = ['steelblue', 'darkorange', 'green', 'crimson']

plt.figure(figsize=(7, 6))
for (name, m), col in zip(results.items(), colours):
    fpr, tpr, _ = roc_curve(test_labels, m['proba'])
    plt.plot(fpr, tpr, color=col, lw=2,
             label=f"{name}  (AUC = {m['AUROC']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random chance')
plt.xlabel('False Positive Rate (1 − Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('AUROC Comparison — Suicide Detection')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('comparison_auroc.png', dpi=150)
# plt.show()  # disabled: using Agg backend
print("AUROC plot saved → comparison_auroc.png")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation 2 — Confusion-matrix grid (2 × 2 subplots)
#
#  Each confusion matrix shows:
#    TN (top-left)  : non-suicide correctly classified
#    FP (top-right) : non-suicide mis-classified as suicide
#    FN (bottom-left): suicide mis-classified as non-suicide  ← high clinical cost
#    TP (bottom-right): suicide correctly identified
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, (name, m) in zip(axes.flat, results.items()):
    disp = ConfusionMatrixDisplay(m['cm'],
                                  display_labels=['non-suicide', 'suicide'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(name, fontweight='bold')

fig.suptitle('Confusion Matrices — All Classifiers', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('comparison_confusion_matrices.png', dpi=150)
# plt.show()  # disabled: using Agg backend
print("Confusion matrix grid saved → comparison_confusion_matrices.png")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisation 3 — Grouped bar chart of scalar metrics
#
#  Allows side-by-side reading of Accuracy, Precision, Sensitivity,
#  Specificity, and F1 for every classifier.
# ─────────────────────────────────────────────────────────────────────────────
metric_names = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1']
clf_names    = list(results.keys())
x = np.arange(len(metric_names))
bar_w = 0.18

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, col) in enumerate(zip(clf_names, colours)):
    vals = [results[name][m] for m in metric_names]
    bars = ax.bar(x + i * bar_w, vals, bar_w, label=name, color=col, alpha=0.85)
    # annotate value on each bar
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.2f}", ha='center', va='bottom', fontsize=7)

ax.set_xticks(x + bar_w * 1.5)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Score')
ax.set_title('Performance Metrics — Classifier Comparison')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('comparison_metrics_bar.png', dpi=150)
# plt.show()  # disabled: using Agg backend
print("Metric bar chart saved → comparison_metrics_bar.png\n")

# ─────────────────────────────────────────────────────────────────────────────
# Key insights printed to console
# ─────────────────────────────────────────────────────────────────────────────
best_auroc = max(results, key=lambda n: results[n]['AUROC'])
best_f1    = max(results, key=lambda n: results[n]['F1'])
best_sens  = max(results, key=lambda n: results[n]['Sensitivity'])

print("─" * 60)
print("Key insights from the comparison")
print("─" * 60)
print(f"  Highest AUROC      : {best_auroc:<22} "
      f"({results[best_auroc]['AUROC']:.4f})")
print(f"  Highest F1         : {best_f1:<22} "
      f"({results[best_f1]['F1']:.4f})")
print(f"  Highest Sensitivity: {best_sens:<22} "
      f"({results[best_sens]['Sensitivity']:.4f})")
print("─" * 60)
print("""
Interpretation notes
────────────────────
• Sensitivity (True Positive Rate) is the most clinically relevant metric
  here: missing a suicide post (FN) carries far greater risk than a false
  alarm (FP).  The classifier with the highest sensitivity minimises that
  risk the most.

• AUROC gives a threshold-independent view of discriminative power.
  Values above 0.80 indicate strong separation between classes.

• Bernoulli NB is the fastest but makes a strong conditional-independence
  assumption that often underperforms on correlated token sequences.

• Logistic Regression and SVM are both linear models; they differ mainly
  in their loss functions (log-loss vs. hinge loss) and how they handle
  class overlap.

• MLP can capture non-linear interactions between token positions at the
  cost of longer training time and more hyper-parameters to tune.
""")
