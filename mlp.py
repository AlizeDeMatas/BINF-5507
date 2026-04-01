import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from collections import Counter

# -------------------------------------------------------
# FILES — update these to match your tokenized CSVs
# e.g. after running 2_bert_tokenize.py on sample_100 and sample_20
# -------------------------------------------------------
train_file = 'sample_100_tokens.csv'
test_file  = 'sample_20_tokens.csv'

# -------------------------------------------------------
# Load data
# -------------------------------------------------------
train_df = pd.read_csv(train_file)
test_df  = pd.read_csv(test_file)

train_labels    = train_df['classification_int']
max_train_count = train_df['max_count'][0]
train_tokens    = train_df['tokens']

test_labels     = test_df['classification_int']
test_text       = test_df['notesCleaned']
max_test_count  = test_df['max_count'][0]
test_tokens     = test_df['tokens']

padding = max(max_train_count, max_test_count)

train_matrix = utils.create_matrix(train_tokens, padding)
test_matrix  = utils.create_matrix(test_tokens, padding)

# -------------------------------------------------------
# MLP Classifier
# -------------------------------------------------------
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(5, 2),
    max_iter=300,
    activation='relu',
    solver='adam',
    random_state=0
)

mlp_clf.fit(train_matrix, train_labels)
predictions = mlp_clf.predict(test_matrix)

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
cm = confusion_matrix(test_labels, predictions)

# Derive metrics from confusion matrix
# cm layout: [[TN, FP], [FN, TP]]
TN, FP, FN, TP = cm.ravel()

precision    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
sensitivity  = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Positive Recall
specificity  = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Negative Recall
f1           = (2 * precision * sensitivity / (precision + sensitivity)
                if (precision + sensitivity) > 0 else 0.0)

print("=" * 50)
print(f"Accuracy:                    {accuracy_score(test_labels, predictions)*100:.2f}%")
print(f"Precision:                   {precision:.4f}")
print(f"Sensitivity / Pos. Recall:   {sensitivity:.4f}")
print(f"Specificity / Neg. Recall:   {specificity:.4f}")
print(f"F1 Score:                    {f1:.4f}")
print("=" * 50)
print("\nConfusion Matrix:")
print(cm)
print("\nFull Classification Report:")
print(classification_report(test_labels, predictions, target_names=['non-suicide', 'suicide']))

# -------------------------------------------------------
# Top 10 indicative words of suicide
# (words most associated with posts predicted as suicide)
# -------------------------------------------------------
def get_indicative_words(texts, preds, n=10):
    word_counts = Counter()
    for text, pred in zip(texts, preds):
        if pred == 1:
            words = str(text).lower().split()
            word_counts.update(words)
    return word_counts.most_common(n)

top_words = get_indicative_words(test_text, predictions)
print("\nTop 10 indicative words of suicide (predicted positive posts):")
for word, count in top_words:
    print(f"  {word}: {count}")

# -------------------------------------------------------
# Plot confusion matrix
# -------------------------------------------------------
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['non-suicide', 'suicide'])
disp.plot(cmap='Blues')
plt.title('MLP Confusion Matrix')
plt.tight_layout()
plt.savefig('mlp_confusion_matrix.png', dpi=150)
plt.show()
print("\nConfusion matrix plot saved to mlp_confusion_matrix.png")
