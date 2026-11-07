# Spam Email Classification (UCI Spambase)

Machine Learning, Assignment 2 (M.Tech AIML, BITS Pilani WILP)

- **Live app:** https://2026ak05376-bits-ml-assignment-2.streamlit.app/
- **GitHub repository:** https://github.com/djhemath/bits-ml-assignment-2

## a. Problem statement
A classic email spam classification problem. Classify an email as **spam** or **not spam** from features that describe it's content. All features are numeric and continous.

These features represent ideas like how often certain words and characters appear, and how long runs of capital letters are.

Trained five classification models and compared with six evaluation metrics such as,
1. Accuracy
2. AUC
3. Precision
4. Recall
5. F1
6. Matthews Correlation Coefficient

These trained models are served through a Streamlit web app where we can perform the inference.


## b. Dataset description

| Property | Value |
|---|---|
| Name | Spambase |
| Source | UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/94/spambase |
| Task | Binary classification (`is_spam`: 1 = spam, 0 = not spam) |
| Raw size | 4,601 rows, 57 features + 1 target |
| After removing exact duplicates | 4,210 rows (391 duplicates dropped); 1,679 spam (39.9%) |
| Features | 57, all continuous and non-negative |
| Missing values | None |

Assignment minimums are 12 features and 500 instances. This dataset has 57 features and 4,210 instances.


**Features (57)**
- 48 `word_freq_*` features: percentage of words in the email that match a given word (for example `word_freq_free`, `word_freq_money`, `word_freq_george`).
- 6 `char_freq_*` features: percentage of characters that match `;`, `(`, `[`, `!`, `$` and `#`. In the code these are named `char_freq_semicolon`, `char_freq_paren`, `char_freq_bracket`, `char_freq_exclamation`, `char_freq_dollar` and `char_freq_hash`.
- 3 capital-letter features: `capital_run_length_average`, `capital_run_length_longest` and `capital_run_length_total`.

**Preparation**
- Exact duplicate rows were removed before splitting, so the same email cannot appear in both the training and test sets.
- Stratified 80/20 train/test split (`random_state = 2026`): 3,368 training rows and 842 test rows (336 spam, 506 not spam). `test_data.csv` is this exact test split.
- Almost every feature is heavily right-skewed. Logistic Regression, KNN and SVM use `log(1 + x)` followed by standardisation. Naive Bayes uses `log(1 + x)` only. The Decision Tree uses the raw features, since a monotonic transform does not change its splits.
- Preprocessing is part of each scikit-learn `Pipeline`, so it is fitted on training data only and saved together with each model.
- Hyperparameters were tuned with 5-fold stratified `GridSearchCV` on ROC AUC, using the training set only. The test set was used once, for the final evaluation.


## c. GitHub repository link

https://github.com/djhemath/bits-ml-assignment-2

## d. Models used

Evaluation is done with a test set of 842 emails.
Spam is the positive class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9264 | 0.9733 | 0.9308 | 0.8810 | 0.9052 | 0.8459 |
| Decision Tree | 0.8979 | 0.9611 | 0.9195 | 0.8155 | 0.8644 | 0.7865 |
| KNN | 0.9299 | 0.9738 | 0.9342 | 0.8869 | 0.9099 | 0.8534 |
| Naive Bayes | 0.8527 | 0.9657 | 0.7454 | 0.9583 | 0.8385 | 0.7259 |
| Support Vector Machine | **0.9371** | **0.9795** | **0.9464** | 0.8929 | **0.9188** | **0.8685** |

Note: the Naive Bayes AUC is about 0.96575, so it displays as 0.9657 or 0.9658 depending on the numpy version used to run the code. All other values are identical across environments.

Along with this, hyperparameters are also tuned. Tuned hyperparameters are,

| Model | Best parameters | CV AUC |
|---|---|---|
| Logistic Regression | `C = 100` | 0.9803 |
| Decision Tree | `max_depth = None`, `min_samples_leaf = 40` | 0.9494 |
| KNN | `n_neighbors = 15`, `weights = distance` | 0.9764 |
| Naive Bayes (Gaussian) | `var_smoothing = 0.01` | 0.9616 |
| SVM (RBF kernel) | `C = 100`, `gamma = 0.003` | 0.9833 |

### Observations
False alarms are wrongly flagged legitimate emails. Misses are spam emails that weren't caught.

This test set has 506 legitimate emails and 336 spam emails

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A strong linear baseline. Accuracy 0.926 and MCC 0.846. It made 22 false alarms and missed 40 spam emails. It performed best with weak regularisation (`C = 100`). It is the simplest model to train and explain, and it is only slightly behind KNN and SVM. |
| Decision Tree | The weakest on AUC (0.961) and recall (0.816). It missed 62 spam emails, the most of any model, with 24 false alarms. Its CV AUC (0.949) was also the lowest. It needed a large `min_samples_leaf` of 40 to generalise, which suggests that a single tree overfits this 57-feature data. |
| KNN | Slightly ahead of Logistic Regression. Accuracy 0.930, AUC 0.974 and MCC 0.853, with 21 false alarms and 38 misses. Distance weighting with 15 neighbours worked best. Because it is distance-based, it depends on the `log(1 + x)` and scaling steps, which stop large-valued features from dominating the distance. |
| Naive Bayes | It has a very different profile. It has the highest recall (0.958) and missed only 14 spam emails, but its precision is the lowest (0.745) because it flagged 110 legitimate emails as spam. That makes its accuracy (0.853), F1 (0.839) and MCC (0.726) the lowest. Its AUC (0.966) is respectable, so it ranks emails reasonably well, but the default 0.5 threshold suits it poorly. The independence assumption does not hold here, since many word frequencies are correlated. |
| Support Vector Machine | The best on accuracy (0.937), AUC (0.980), precision (0.946), F1 (0.919) and MCC (0.869). The RBF kernel captures non-linear structure that Logistic Regression cannot. It made the fewest false alarms (17) and the second-fewest misses (36). |

**Overall winner for this dataset: Support Vector Machine (RBF kernel).** It leads on five of the six metrics and has the best mean rank across all six. Naive Bayes wins only on recall. KNN and Logistic Regression are close behind and are cheaper to train and easier to explain. The Decision Tree and Naive Bayes are the weakest overall, for different reasons: the tree misses spam, while Naive Bayes over-flags legitimate mail. For a spam filter, a false alarm is usually costlier than a missed spam email, which favours a high-precision model such as the SVM.

Results are for `random_state = 2026`. Exact figures may shift slightly with a different split, but the overall ranking pattern should hold.


## Streamlit app

The app loads the five saved models from `model/` and evaluates them on a test CSV. It never retrains.

- Upload a test CSV, download a sample, or click **Load sample data** to use the bundled `test_data.csv`.
- Choose a model from a dropdown.
- View Accuracy, AUC, Precision, Recall, F1 and MCC.
- View the confusion matrix and classification report for the selected model.
- Compare all five models on the same file, with a metrics table, bar chart and ROC curves.
- Adjust the decision threshold for precision, recall, F1, MCC and the confusion matrix. AUC does not depend on the threshold.

The CSV must contain the 57 feature columns and an `is_spam` column (1 = spam, 0 = not spam). Extra columns are ignored. Files with missing columns, missing values, non-numeric values, negative feature values, or labels other than 0 and 1 are rejected with a message.


## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

To retrain and regenerate `model/` and `test_data.csv`, run all cells of `spambase_classification.ipynb` from the repository root. The saved models are tied to the scikit-learn version in `requirements.txt` (1.5.x). If you retrain under a different version, commit the new `model/` files.
