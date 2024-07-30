import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
KAGGLE_KEY = os.getenv('KAGGLE_KEY')

# Ensure Kaggle API key is available
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
    f.write(f'{{"username":"{KAGGLE_USERNAME}","key":"{KAGGLE_KEY}"}}')

# Authenticate Kaggle API
api = KaggleApi()
api.authenticate()

# Define paths
data_path = 'data'
competition_name = 'playground-series-s4e7'
zip_file = f'{data_path}/{competition_name}.zip'
train_file = f'{data_path}/train.csv'
test_file = f'{data_path}/test.csv'

# Download the dataset if not already present
if not os.path.exists(zip_file):
    api.competition_download_files(competition_name, path=data_path)

# Extract the dataset if not already extracted
if not os.path.exists(train_file) or not os.path.exists(test_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(data_path)

# Load the data
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

# Encode categorical features
label_encoders = {}
for column in train.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train[column] = le.fit_transform(train[column])
    test[column] = le.transform(test[column])
    label_encoders[column] = le

# Fill missing values
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)

# Define target column
target_column = 'Response'

# Split features and target
X = train.drop(target_column, axis=1)
y = train[target_column]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to train and evaluate models
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    print(f'{model_name} F1 Score: {f1}')
    return model, f1

# Train and evaluate Random Forest model
rf_model, rf_f1 = train_and_evaluate(RandomForestClassifier(class_weight='balanced', n_jobs=-1), 'Random Forest')

# Train and evaluate XGBoost model
xgb_model, xgb_f1 = train_and_evaluate(xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), n_jobs=-1, use_label_encoder=False, eval_metric='logloss'), 'XGBoost')

# Train and evaluate LightGBM model
lgb_model, lgb_f1 = train_and_evaluate(lgb.LGBMClassifier(scale_pos_weight=(y == 0).sum() / (y == 1).sum(), n_jobs=-1), 'LightGBM')

# Select the best model based on validation F1 score
best_model = max([(rf_model, rf_f1), (xgb_model, xgb_f1), (lgb_model, lgb_f1)], key=lambda item: item[1])[0]

# Make predictions on the test set with the best model
y_test_pred = best_model.predict(test.drop('id', axis=1))
submission = pd.DataFrame({'Id': test['id'], target_column: y_test_pred})
submission.to_csv('submission.csv', index=False)
