# Loan Prediction Dashboard

**Live demo:** https://loan-prediction-shavali.streamlit.app

> Open the link, choose **Bank Loan Approval Prediction**, and upload the `train.csv` file from the Kaggle Loan Prediction dataset to reproduce the results below.

A Streamlit web app that trains and evaluates machine learning models on the Kaggle Loan Prediction dataset. Users can upload the data, compare models, save the best one, and predict on new applicants.

## Features

- Explore the dataset: row counts, missing values, and summary statistics
- Train Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, or SVM
- Evaluate with accuracy, precision, recall, F1, ROC-AUC, a confusion matrix, and an ROC curve
- Cross-validation and GridSearchCV tuning for an SVM
- Save a trained model with joblib, reload it, and predict on new data

## Tech Stack

Python, scikit-learn, pandas, NumPy, Streamlit, matplotlib, joblib, pytest

## Results (Loan Approval)

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.862 | 0.840 | 0.988 | 0.852 |
| Random Forest | 0.821 | 0.846 | 0.906 | 0.786 |

Logistic Regression had the best overall performance. Its 5-fold cross-validation accuracy was 0.796 ± 0.043.

**What I learned:** Logistic Regression performed best on this dataset, with high recall, which matters when missing an eligible applicant is costly. Random Forest had slightly higher precision. The cross-validation score is lower than the test accuracy, so the result depends somewhat on how the data is split. Next, I would test more models and tune them.

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 and upload `train.csv` from the Kaggle Loan Prediction dataset.

## Run Tests

```bash
python -m pytest -v
```

## Author

**Shaik Shekshavali**: B.Tech CSE-IT, Vasireddy Venkatadri Institute of Technology, Guntur
- LinkedIn: https://www.linkedin.com/in/shaik-shekshavali
- GitHub: https://github.com/shaik-shekshavali
