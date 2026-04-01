# BINF-5507-Suicide-Detection

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

### How to Run

**1. Clone the repo**
```bash
git clone https://github.com/AlizeDeMatas/BINF-5507.git
cd BINF-5507
```

**2. Set up the environment**
```bash
conda env create -f environment.yml
conda activate suicide-detection
```
Or with pip:
```bash
pip install -r requirements.txt
```

**3. Download NLTK data**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
```

**4. Tokenize the data**

Open `2_bert_tokenize.py` and set `name = 'sample_100'` on line 8, then run:
```bash
python 2_bert_tokenize.py
```
Then change `name = 'sample_20'` and run again:
```bash
python 2_bert_tokenize.py
```
This generates `sample_100_tokens.csv` and `sample_20_tokens.csv`.

**5. Run the MLP classifier**
```bash
python mlp.py
```
Results will print to the terminal and a confusion matrix plot will be saved as `mlp_confusion_matrix.png`.
