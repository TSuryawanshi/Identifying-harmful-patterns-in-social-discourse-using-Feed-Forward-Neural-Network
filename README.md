# Final Hate Speech Classification Project

This project builds a machine learning pipeline for detecting hate, offensive, and neutral speech in short-form text. It includes dataset processing, cleaning, balancing, TF-IDF vectorization, model training, and an example inference step.

---

## 📁 Project Structure

```
.
├── main.py
├── hate_speech.csv
├── cleaned_hate_dataset.csv        # (generated)
└── hate_speech_model.pkl           # (generated)
```

---

## 🔍 Features

* Cleans and normalizes tweet text
* Removes stopwords, URLs, mentions, emojis, and special chars
* Handles class imbalance via oversampling
* Trains a neural network classifier (MLP)
* Uses TF-IDF embeddings with n-grams
* Saves trained model + vectorizer as artifacts
* Contains inference demo for quick testing

---

## 🧠 Model Architecture

**Algorithm:** MLPClassifier (sklearn)

**Key Params**

* `hidden_layer_sizes=(100,)`
* `activation="relu"`
* `solver="adam"`
* `max_iter=200`
* `early_stopping=True`

---

## 🗂 Dataset Details

Classes:

* `0` → Hate
* `1` → Offensive
* `2` → Neutral

The script balances the dataset by oversampling under-represented labels before training.

---

## 🧹 Text Cleaning Pipeline

The `clean_text()` function applies:

* Lowercasing
* URL + username removal
* Hashtag stripping
* Punctuation removal
* Stopword filtering
* Token reconstruction

Example input:

```
"@user Check out this link! http://site.com #Wow"
```

Output:

```
"check link"
```

---

## 🚀 Running the Project

### 1. Install dependencies

```
pip install -r requirements.txt
```

(If NLTK stopwords aren't installed, script will download them.)

### 2. Execute training

```
python main.py
```

This will:

* Load dataset
* Clean + balance
* Train model
* Save artifacts
* Display metrics

---

## 📦 Output Artifacts

After training completes:

* `cleaned_hate_dataset.csv`
* `hate_speech_model.pkl`

The `.pkl` file stores:

```python
{
  "vectorizer": TfidfVectorizer,
  "model": MLPClassifier
}
```

---

## 🧪 Example Prediction

The script ends by running inference on sample inputs such as:

```
"I hate you so much!"
"Check out my new blog post"
"Had a great time at the concert!"
```

Outputs include predicted class labels.

---

## 🧰 Requirements

Recommended environment:

* Python 3.8+
* scikit-learn
* imbalanced-learn
* pandas
* nltk

---

## 📊 Evaluation

Model prints:

* Classification report
* Test accuracy
* Balanced metrics for all classes

Example metric snippet:

```
precision / recall / f1-score
Test Accuracy: X.XXXX
```

---

## 📬 Notes

* Designed for academic + research use
* Not production-hardened
* Requires dataset `hate_speech.csv` to exist locally

---
