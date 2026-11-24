# Titanic Survival Prediction

A machine learning project that predicts whether a Titanic passenger survived, and compares 5 classification models.

## Dataset
Titanic dataset from Seaborn (891 passengers, 889 after removing 2 rows with a missing embarkation port).

Features used: passenger class, sex, age, siblings/spouses aboard, parents/children aboard, fare, port of embarkation, and whether the passenger was alone.

## Data Preparation
- Dropped columns that repeat other information (deck, embark_town, alive, class, who, adult_male)
- Filled missing ages with the median
- Encoded sex and embarked with LabelEncoder
- Scaled features with StandardScaler (inside a Pipeline, so there is no data leakage)

## Results (5-fold cross-validation)
| Model | Accuracy |
|---|---|
| SVM | 82.7% |
| KNN | 80.1% |
| Logistic Regression | 78.7% |
| Naive Bayes | 78.4% |
| Decision Tree | 77.6% |

## What I Learned
- Fixed mistakes in my first version: comparing predictions with the wrong variable, and fitting the scaler on test data.
- Using a Pipeline keeps scaling inside cross-validation and gives a more honest comparison than a single train/test split.
- A single Decision Tree overfits easily and scored lowest.

## Tech Stack
Python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib

## How to Run
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
jupyter notebook titanic_survival_prediction.ipynb
```
