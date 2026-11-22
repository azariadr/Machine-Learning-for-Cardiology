# Echocardiogram Survival Prediction Pipeline

This repository contains a robust, leak-free machine learning pipeline designed to predict patient survival (`StillAlive`) using clinical echocardiogram data from the UCI Machine Learning Repository.

## Project Overview

The objective is to predict post-heart attack survival using clinical metrics from an echocardiogram (ultrasound of the heart). 

To ensure a production-grade, statistically sound evaluation, the project addresses two major challenges in clinical ML workflows:
1. **Target Leakage**: Removing variables like `Survival` (which directly relates to the outcome window) that would not be safely known in a real-world pre-assessment setting.
2. **Data Leakage**: Wrapping preprocessing steps like imputation and scaling inside a scikit-learn `Pipeline` to ensure statistics from the validation/test folds never influence the training folds during validation.

## Data Pipeline Architecture

```
[Raw Echocardiogram Data] 
       └───> Drop Missing Targets & Non-Clinical Features
                 └───> Separate Features (X) & Target (y) [Dropping Target Leakage: 'Survival']
                           └───> Repeated Stratified K-Fold Cross-Validation (5x5)
                                     └───> Pipeline Evaluation:
                                               ├── Imputation (Median) [Folds-Isolated]
                                               ├── Scaling (StandardScaler)
                                               └── Classification Algorithms
```

### Key Preprocessing Steps (No-Leakage Policy):
- **Numerical Features**: Imputed using the `median` strategy and normalized using `StandardScaler` inside the pipeline folds.
- **Categorical Features**: Handled as pass-through indicators.
- **Cross-Validation**: Evaluated using **Repeated Stratified K-Fold (5 splits, 5 repeats)** to obtain stable, reliable validation statistics with tight confidence intervals.

## Models Evaluated

We benchmarked a variety of algorithms, ranging from basic baselines to tuned ensembles:
- Baseline (Most Frequent Classifier)
- Logistic Regression (with class weight balancing)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- **Tuned XGBoost** (Optimized via `RandomizedSearchCV` wrapped inside the nested cross-validation fold loops to avoid optimization leakage)

##  Results

The models were evaluated across multiple metrics, including Accuracy, F1-Score, and Area Under the ROC Curve (ROC-AUC):

| Model | Accuracy (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **72.07** | **59.66** | **76.56** |
| Random Forest | 70.09 | 53.51 | 74.65 |
| Tuned XGBoost | 69.19 | 46.19 | 72.31 |
| Baseline | 67.18 | 0.00 | 50.00 |
| Gradient Boosting | 65.65 | 43.68 | 66.46 |
| Decision Tree | 61.07 | 50.82 | 62.48 |

*Note: Logistic Regression (balanced) performed best on this clinical dataset, demonstrating high diagnostic power (76.56% ROC-AUC) without overfitting on the small sample size (n=131).* 

## ᎐ How to Use the Notebook

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn
   ```
2. Open and run the Jupyter notebook to load data directly from the UCI ML Repository and train the pipeline.
