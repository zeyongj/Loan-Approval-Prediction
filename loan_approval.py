# Install necessary packages
!pip install category_encoders
!pip install optuna
!pip install imbalanced-learn
!pip install lightgbm --upgrade

# Import necessary libraries
import os
import pandas as pd
import numpy as np
import zipfile
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import category_encoders as ce
from imblearn.over_sampling import SMOTE

# Extract the uploaded zip file
zip_file_path = '/content/playground-series-s4e10.zip'
extract_dir = '/content/playground-series-s4e10/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Load the datasets
train_df = pd.read_csv(os.path.join(extract_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(extract_dir, 'test.csv'))
sample_submission_df = pd.read_csv(os.path.join(extract_dir, 'sample_submission.csv'))

# Reset index to 'id' for both datasets
train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)

# Identify categorical and numerical features
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
numerical_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
                      'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']

# Feature Engineering: Create new features
def feature_engineering(df):
    # Income to loan amount ratio
    df['income_loan_ratio'] = df['person_income'] / df['loan_amnt']
    # Employment length to age ratio
    df['emp_age_ratio'] = df['person_emp_length'] / df['person_age']
    # Interest rate to income ratio
    df['int_rate_income_ratio'] = df['loan_int_rate'] / df['person_income']
    # Credit history length to age ratio
    df['cred_hist_age_ratio'] = df['cb_person_cred_hist_length'] / df['person_age']
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

# Update numerical features with new features
numerical_features.extend(['income_loan_ratio', 'emp_age_ratio', 'int_rate_income_ratio', 'cred_hist_age_ratio'])

# Separate features and target
X = train_df.drop(columns='loan_status')
y = train_df['loan_status']

# Initialize Target Encoder
target_enc = ce.TargetEncoder(cols=categorical_features)

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare arrays for out-of-fold predictions
oof_preds = np.zeros(X.shape[0])
test_preds = np.zeros(test_df.shape[0])

# Define objective function for Optuna
def objective(trial):
    aucs = []
    for train_index, valid_index in skf.split(X, y):
        X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
        y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

        # Target Encoding
        X_train_fold = target_enc.fit_transform(X_train_fold, y_train_fold)
        X_valid_fold = target_enc.transform(X_valid_fold)
        X_test_enc = target_enc.transform(test_df)

        # Handle class imbalance with SMOTE
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X_train_fold, y_train_fold)

        # Scale numerical features
        scaler = StandardScaler()
        X_resampled[numerical_features] = scaler.fit_transform(X_resampled[numerical_features])
        X_valid_fold[numerical_features] = scaler.transform(X_valid_fold[numerical_features])
        X_test_enc[numerical_features] = scaler.transform(X_test_enc[numerical_features])

        # Define LightGBM parameters using new suggest methods
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            'random_state': 42,
            'verbosity': -1,
            'n_jobs': -1
        }

        lgb_train = lgb.Dataset(X_resampled, y_resampled)
        lgb_valid = lgb.Dataset(X_valid_fold, y_valid_fold, reference=lgb_train)

        # Use early_stopping as a callback
        pruning_callback = LightGBMPruningCallback(trial, 'auc')
        early_stopping_callback = lgb.early_stopping(stopping_rounds=100, verbose=False)
        callbacks = [pruning_callback, early_stopping_callback]

        gbm = lgb.train(param,
                        lgb_train,
                        num_boost_round=10000,
                        valid_sets=[lgb_train, lgb_valid],
                        verbose_eval=False,
                        callbacks=callbacks)

        y_valid_pred = gbm.predict(X_valid_fold, num_iteration=gbm.best_iteration)
        auc = roc_auc_score(y_valid_fold, y_valid_pred)
        aucs.append(auc)

    return np.mean(aucs)

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction='maximize', study_name='lgbm_classifier')
study.optimize(objective, n_trials=50)

# Retrieve the best parameters
best_params = study.best_params
best_params['objective'] = 'binary'
best_params['metric'] = 'auc'
best_params['boosting_type'] = 'gbdt'
best_params['random_state'] = 42
best_params['verbosity'] = -1
best_params['n_jobs'] = -1

print('Best Hyperparameters:')
print(best_params)

# Train the model with best hyperparameters and make predictions
for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print(f'Fold {fold + 1}')
    X_train_fold, X_valid_fold = X.iloc[train_index], X.iloc[valid_index]
    y_train_fold, y_valid_fold = y.iloc[train_index], y.iloc[valid_index]

    # Target Encoding
    X_train_fold = target_enc.fit_transform(X_train_fold, y_train_fold)
    X_valid_fold = target_enc.transform(X_valid_fold)
    X_test_enc = target_enc.transform(test_df)

    # Handle class imbalance with SMOTE
    sm = SMOTE(random_state=42)
    X_resampled, y_resampled = sm.fit_resample(X_train_fold, y_train_fold)

    # Scale numerical features
    scaler = StandardScaler()
    X_resampled[numerical_features] = scaler.fit_transform(X_resampled[numerical_features])
    X_valid_fold[numerical_features] = scaler.transform(X_valid_fold[numerical_features])
    X_test_enc[numerical_features] = scaler.transform(X_test_enc[numerical_features])

    lgb_train = lgb.Dataset(X_resampled, y_resampled)
    lgb_valid = lgb.Dataset(X_valid_fold, y_valid_fold, reference=lgb_train)

    # Use early_stopping as a callback
    early_stopping_callback = lgb.early_stopping(stopping_rounds=100, verbose=False)
    callbacks = [early_stopping_callback]

    gbm = lgb.train(best_params,
                    lgb_train,
                    num_boost_round=10000,
                    valid_sets=[lgb_train, lgb_valid],
                    verbose_eval=100,
                    callbacks=callbacks)

    # Predict on validation set
    y_valid_pred = gbm.predict(X_valid_fold, num_iteration=gbm.best_iteration)
    oof_preds[valid_index] = y_valid_pred

    # Predict on test set
    test_fold_pred = gbm.predict(X_test_enc, num_iteration=gbm.best_iteration)
    test_preds += test_fold_pred / skf.n_splits

# Evaluate the overall model performance
roc_auc = roc_auc_score(y, oof_preds)
print(f'Overall ROC-AUC Score: {roc_auc}')

# Prepare the submission file
submission_df = pd.DataFrame({'id': test_df.index, 'loan_status': test_preds})
submission_file_path = '/content/loan_approval_submission_optimized.csv'
submission_df.to_csv(submission_file_path, index=False)

# Display the first few rows of the submission file
print(submission_df.head())
