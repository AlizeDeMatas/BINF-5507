# -*- coding: utf-8 -*-
"""
wordclouds.py — BINF-5507 Assignment
Generates word clouds split by prediction outcome:
  True Positive  (TP) — predicted suicide,     actually suicide       ✓ correct
  True Negative  (TN) — predicted non-suicide, actually non-suicide   ✓ correct
  False Positive (FP) — predicted suicide,     actually non-suicide   ✗ false alarm
  False Negative (FN) — predicted non-suicide, actually suicide       ✗ missed case

Why word clouds per outcome?
  - TPs show the vocabulary that reliably signals suicide risk.
  - FNs reveal language the model fails to flag (subtle / indirect phrasing).
  - FPs expose words that mislead the model into false alarms.
  - TNs provide a baseline of non-risk language for contrast.

Datasets used
  Training : datasets/sample_100_tokens1.csv  (92 posts, BERT-tokenized)
  Test     : 500-post random sample from
             datasets/reddit_depression_suicidewatch_tokens1.csv
             (r/SuicideWatch = suicide, r/depression = non-suicide)
  The random sample is large enough to populate all four outcome groups
  while keeping matrix construction fast.

Run
  python wordclouds.py
Output
  wordcloud_TP.png, wordcloud_TN.png, wordcloud_FP.png, wordcloud_FN.png
  wordclouds_grid.png  — all four in one figure
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # save PNGs without opening display windows

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import utils  # project-local helper — create_matrix()

from sklearn.neural_network import MLPClassifier

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load training data
#     Same 92-post sample used in assignment_1_4_and_1_5.py so that the
#     trained MLP is identical and results are directly comparable.
# ─────────────────────────────────────────────────────────────────────────────
train_df     = pd.read_csv('datasets/sample_100_tokens1.csv', encoding='latin-1')
train_labels = train_df['classification_int']
max_train    = train_df['max_count'].iloc[0]   # 1964 — longest post in training set

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load test data — 500-post random sample from the Reddit dataset
#
#     The full Reddit file has 19,257 posts but building a 19k × 6554 matrix
#     requires ~1 GB of RAM.  A 500-post sample gives roughly:
#       ~230 TP, ~210 TN, and some FP / FN — more than enough for word clouds.
#
#     random_state=42 makes the sample reproducible.
#     We cap token sequences at max_train (1964) so the feature matrix
#     stays the same width as training — posts longer than 1964 tokens are
#     simply truncated (the tail tokens are dropped).
# ─────────────────────────────────────────────────────────────────────────────
full_test_df = pd.read_csv(
    'datasets/reddit_depression_suicidewatch_tokens1.csv',
    encoding='latin-1'
)
test_df     = full_test_df.sample(n=500, random_state=42).reset_index(drop=True)
test_labels = test_df['classification_int']
test_text   = test_df['notesCleaned']   # cleaned text used for word cloud content

# Use training padding (1964) — test posts longer than this are truncated
padding = max_train

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Build feature matrices
#     utils.create_matrix parses the stored token-ID strings into a
#     zero-padded numpy array of shape (n_posts, padding).
# ─────────────────────────────────────────────────────────────────────────────
print("Building feature matrices...")
train_matrix = utils.create_matrix(train_df['tokens'], padding)
test_matrix  = utils.create_matrix(test_df['tokens'],  padding)
print(f"  train: {train_matrix.shape},  test: {test_matrix.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Train MLP — same hyper-parameters as assignment_1_4_and_1_5.py
# ─────────────────────────────────────────────────────────────────────────────
print("Training MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    max_iter=500,
    random_state=42,
)
mlp.fit(train_matrix, train_labels)
predictions = mlp.predict(test_matrix)
print("  Done.\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Split posts into outcome groups
#
#     actual=1 (suicide),  predicted=1  →  True Positive  (TP)
#     actual=0 (non-sui),  predicted=0  →  True Negative  (TN)
#     actual=0 (non-sui),  predicted=1  →  False Positive (FP)  ← false alarm
#     actual=1 (suicide),  predicted=0  →  False Negative (FN)  ← missed case
# ─────────────────────────────────────────────────────────────────────────────
actual = test_labels.values

tp_mask = (actual == 1) & (predictions == 1)
tn_mask = (actual == 0) & (predictions == 0)
fp_mask = (actual == 0) & (predictions == 1)
fn_mask = (actual == 1) & (predictions == 0)

groups = {
    'True Positive (TP)\nPredicted suicide — actually suicide':
        test_text[tp_mask],
    'True Negative (TN)\nPredicted non-suicide — actually non-suicide':
        test_text[tn_mask],
    'False Positive (FP)\nPredicted suicide — actually non-suicide':
        test_text[fp_mask],
    'False Negative (FN)\nPredicted non-suicide — actually suicide':
        test_text[fn_mask],
}

print("Outcome group sizes:")
labels_short = ['TP', 'TN', 'FP', 'FN']
masks        = [tp_mask, tn_mask, fp_mask, fn_mask]
for label, mask in zip(labels_short, masks):
    print(f"  {label}: {mask.sum()} posts")
print()

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Build word clouds
#
#     STOPWORDS removes common English function words (the, and, is …) so
#     the clouds show only content-bearing words.
#     Additional domain stop-words are added to suppress generic Reddit noise.
#
#     WordCloud settings:
#       max_words=80    — show the 80 most frequent words per cloud
#       collocations=False — count each word independently (no bigrams)
#       background_color='white' — clean look for reports
# ─────────────────────────────────────────────────────────────────────────────
extra_stopwords = {
    'like', 'just', 'know', 'want', 'dont', 'feel', 'really', 'get',
    'got', 'one', 'would', 'even', 'im', 'ive', 'people', 'think',
    'make', 'time', 'going', 'things', 'thing', 'much', 'still',
    'never', 'always', 'life', 'day', 'way', 'good', 'bad', 'need',
    'see', 'something', 'anyone', 'everything', 'nothing', 'doesnt',
    'cant', 'could', 'lot', 'also', 'back', 'ever', 'every', 'years',
    'year', 'long', 'little', 'used', 'amp',
}
stop = STOPWORDS.union(extra_stopwords)

colours = {
    'True Positive (TP)\nPredicted suicide — actually suicide':       'Reds',
    'True Negative (TN)\nPredicted non-suicide — actually non-suicide': 'Blues',
    'False Positive (FP)\nPredicted suicide — actually non-suicide':  'Oranges',
    'False Negative (FN)\nPredicted non-suicide — actually suicide':  'Purples',
}

file_names = {
    'True Positive (TP)\nPredicted suicide — actually suicide':       'wordcloud_TP.png',
    'True Negative (TN)\nPredicted non-suicide — actually non-suicide': 'wordcloud_TN.png',
    'False Positive (FP)\nPredicted suicide — actually non-suicide':  'wordcloud_FP.png',
    'False Negative (FN)\nPredicted non-suicide — actually suicide':  'wordcloud_FN.png',
}

generated = {}   # store WordCloud objects for the grid plot

for title, texts in groups.items():
    corpus = ' '.join(texts.dropna().astype(str))

    if len(corpus.strip()) < 10:
        print(f"  Skipping {title.splitlines()[0]} — not enough text ({len(texts)} posts).")
        generated[title] = None
        continue

    wc = WordCloud(
        width=800,
        height=500,
        max_words=80,
        background_color='white',
        colormap=colours[title],
        stopwords=stop,
        collocations=False,
    ).generate(corpus)

    generated[title] = wc

    # Save individual PNG
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(file_names[title], dpi=150)
    plt.close()
    print(f"  Saved {file_names[title]}")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Combined 2×2 grid figure
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 11))
fig.suptitle(
    'Word Clouds by MLP Prediction Outcome\n'
    'Training: sample_100 (92 posts)  |  Test: 500-post Reddit sample',
    fontsize=14, fontweight='bold', y=1.01
)

for ax, (title, wc) in zip(axes.flat, generated.items()):
    if wc is None:
        ax.text(0.5, 0.5, f"No posts\n{title.splitlines()[0]}",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.axis('off')
    else:
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('wordclouds_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nGrid saved → wordclouds_grid.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Print top words per group (useful for written analysis)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Top 15 words per outcome group")
print("=" * 60)
for title, texts in groups.items():
    corpus = ' '.join(texts.dropna().astype(str))
    if len(corpus.strip()) < 10:
        continue
    wc_freq = WordCloud(stopwords=stop, collocations=False,
                        max_words=15).generate(corpus)
    top_words = sorted(wc_freq.words_.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"\n{title.splitlines()[0]}")
    print("  " + ", ".join(f"{w}({round(s,2)})" for w, s in top_words))

print("""
─────────────────────────────────────────────────────────────
How to interpret the word clouds
─────────────────────────────────────────────────────────────
TRUE POSITIVES (red)
  Words the model correctly associated with suicide risk.
  Expect direct language: suicide-related terms, expressions
  of hopelessness, self-harm, and pain.

TRUE NEGATIVES (blue)
  Vocabulary of posts correctly identified as non-suicidal.
  Typically everyday emotional language — relationship issues,
  anxiety, depression without explicit suicidal content.

FALSE POSITIVES (orange)
  Non-suicide posts that triggered the model.
  These words look superficially similar to suicide-risk
  language — the model over-generalised from training.
  Key insight: which emotional words caused false alarms?

FALSE NEGATIVES (purple)
  Suicide posts the model missed.
  Often indirect, coded, or understated language that does
  not match the explicit patterns learned during training.
  Key insight: what vocabulary does the model fail to flag?
─────────────────────────────────────────────────────────────
""")
