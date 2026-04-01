# CPS803-Suicide-Detection

### Abstract
Project uses MLP (Multi-Layer Perceptron) to detect probable suicide messages based on social media posts, distinguishing Reddit posts that indicate suicide and non-suicide.

### MLP Results (sample_100 train / sample_20 test)
Accuracy: 68.42 % <br>
Precision: 0.6842 <br>
Sensitivity / Positive Recall: 1.0000 <br>
Specificity / Negative Recall: 0.0000 <br>
F1 Score: 0.8125 <br>

| | non-suicide | suicide |
|---|---|---|
| Predicted non-suicide | 0 | 0 |
| Predicted suicide | 6 | 13 |

> **Note:** The MLP classifier predicted all test samples as suicide, indicating the model is biased toward the majority class. This is likely due to the small dataset size (100 training samples). Performance is expected to improve significantly with a larger training set.
