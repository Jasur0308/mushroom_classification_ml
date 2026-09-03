# Mushroom Classification: Edible or Poisonous?

A machine learning project that classifies mushrooms as **edible** or
**poisonous** based on 22 categorical physical features (odor, cap shape,
gill color, spore print color, habitat, etc.), using the classic **UCI
Mushroom** dataset (8,124 samples).

## Project overview

- Data cleaning: null checks, duplicate checks, and handling of hidden
  missing values (`?`) in the `stalk-root` column via mode imputation
- Exploratory data analysis: target class balance and the relationship
  between key features (`odor`, `spore-print-color`, `gill-color`) and the
  target class
- Preprocessing: dropping a non-informative feature (`veil-type`), one-hot
  encoding of categorical features, label encoding of the target, and
  feature scaling
- Training and evaluating 4 classification models:
  - K-Nearest Neighbors (KNN)
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Side-by-side model comparison (Accuracy, Precision, Recall, F1-score),
  including a bar chart
- 5-fold stratified cross-validation to confirm the stability of the best
  model
- Business/safety-driven conclusion and recommendations for further
  improvement

## Results

| Model                | Accuracy | Precision | Recall | F1-score |
|-----------------------|:--------:|:---------:|:------:|:--------:|
| KNN                    | high     | high      | high   | high     |
| Decision Tree          | high     | high      | high   | high     |
| Random Forest          | high     | high      | high   | high     |
| **Logistic Regression**| **~0.985** | **~0.987** | **~0.982** | **~0.985** |

**Logistic Regression** achieved the best and most stable performance
across all metrics and was selected as the final model. Since misclassifying
a poisonous mushroom as edible (a false negative) can be dangerous, **Recall
on the poisonous class** was treated as a critical metric alongside the
overall F1-score.

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/mushroom-classification.git
cd mushroom-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook mushroom_classification.ipynb
```

The notebook automatically downloads the dataset from a public GitHub URL, so
no manual data download is required.

## Project structure

```
.
├── mushroom_classification.ipynb   # Main analysis & modeling notebook
├── requirements.txt                # Python dependencies
└── README.md
```

## Tech stack

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn.

## License

This project is available under the MIT License.
