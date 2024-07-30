# Binary Classification of Insurance Cross-Selling

## Project Description
This project aims to predict whether insurance customers will opt for cross-selling insurance products. The task involves binary classification where the target variable indicates whether a customer will purchase an additional product.

## Dataset Information
The dataset consists of customer demographic information, policy details, and previous purchase behaviors. The columns in the dataset are:
- `id`
- `Gender`
- `Age`
- `Driving_License`
- `Region_Code`
- `Previously_Insured`
- `Vehicle_Age`
- `Vehicle_Damage`
- `Annual_Premium`
- `Policy_Sales_Channel`
- `Vintage`
- `Response` (target variable for training data)

## Project Structure
```
├── data
│   ├── train.csv
│   ├── test.csv
├── model.py
├── submission.csv
└── README.md
```

## Setup Instructions
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Ismat-Samadov/Binary-Classification-of-Insurance-Cross-Selling.git
   cd Binary-Classification-of-Insurance-Cross-Selling
   ```

2. **Create a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install OpenMP for XGBoost (macOS):**
   ```bash
   brew install libomp
   ```

## Data Preprocessing
1. **Load the Data:**
   ```python
   train = pd.read_csv('data/train.csv')
   test = pd.read_csv('data/test.csv')
   ```

2. **Encode Categorical Features:**
   ```python
   from sklearn.preprocessing import LabelEncoder
   label_encoders = {}
   for column in train.select_dtypes(include=['object']).columns:
       le = LabelEncoder()
       train[column] = le.fit_transform(train[column])
       test[column] = le.transform(test[column])
       label_encoders[column] = le
   ```

3. **Handle Missing Values:**
   ```python
   train.fillna(train.median(), inplace=True)
   test.fillna(test.median(), inplace=True)
   ```

## Model Training
### Using SMOTE for Handling Imbalanced Data
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)
```

### Random Forest, XGBoost, and LightGBM Models
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train and evaluate models
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f'{model_name} F1 Score: {f1}')
    return model, f1

rf_model, rf_f1 = train_and_evaluate(RandomForestClassifier(class_weight='balanced'), 'Random Forest')
xgb_model, xgb_f1 = train_and_evaluate(xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum()), 'XGBoost')
lgb_model, lgb_f1 = train_and_evaluate(lgb.LGBMClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum()), 'LightGBM')

# Select the best model
best_model = max([(rf_model, rf_f1), (xgb_model, xgb_f1), (lgb_model, lgb_f1)], key=lambda item: item[1])[0]
```

## Model Evaluation
Evaluate the models using the F1 score metric to handle imbalanced data effectively.

## Submission
1. **Make Predictions on Test Set:**
   ```python
   y_test_pred = best_model.predict(test.drop('id', axis=1))
   submission = pd.DataFrame({'Id': test['id'], 'Response': y_test_pred})
   submission.to_csv('submission.csv', index=False)
   ```

2. **Submit to Kaggle:**
   Upload the `submission.csv` file to the Kaggle competition to see your model's performance on the leaderboard.

## Conclusion
This project demonstrates how to handle imbalanced datasets using SMOTE and ensemble models like Random Forest, XGBoost, and LightGBM. The best model is selected based on the validation F1 score and used for making predictions on the test set.

## References
- [Kaggle Competition Page](https://www.kaggle.com/competitions/playground-series-s4e7)
- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [Random Forest Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)