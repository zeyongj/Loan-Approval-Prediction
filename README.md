# Loan Approval Prediction Competition

This repository presents a comprehensive solution for the Kaggle Playground Series competition, Season 4, Episode 10. The objective of the competition was to predict loan approval status using a dataset comprising synthetic features derived from a real-world deep learning model.

## Final Model Performance

Upon submission to Kaggle, the model achieved a final ROC-AUC score of **0.96025**, demonstrating a substantial enhancement over the initial benchmark. This outcome was realized through the integration of advanced techniques, including LightGBM, hyperparameter optimization with Optuna, feature engineering, and addressing data imbalance using SMOTE.

## Files Included

- **`loan_approval_prediction.ipynb`**: The complete Python script used for data preprocessing, model training, and generating predictions for submission, running on Google Colab.
- **`loan_approval_submission_optimized.csv`**: The submission file containing loan approval probabilities for each applicant in the test set.
- **`loan_approval.py`**: The complete Python script used for data preprocessing, model training, and generating predictions for submission, running locally.
- **`README.md`**: This documentation file.

## Installation

To execute this solution, the following Python libraries are required:

```sh
!pip install category_encoders
!pip install optuna
!pip install imbalanced-learn
!pip install lightgbm --upgrade
```

## Dataset Information

The dataset comprises the following files:

- **`train.csv`**: Contains the training data, including features and the target variable (`loan_status`).
- **`test.csv`**: Contains the test data for which loan approval probabilities are to be predicted.
- **`sample_submission.csv`**: Provides the expected format for the submission file.

These files are initially extracted from the provided ZIP file using Python's `zipfile` module.

## Workflow

The solution proceeds through the following steps:

### 1. Feature Engineering

- **Creation of New Features**: Several new features were generated to enhance the model's predictive capability, including `income_loan_ratio`, `emp_age_ratio`, `int_rate_income_ratio`, and `cred_hist_age_ratio`.

### 2. Data Preprocessing

- **Encoding Categorical Features**: `TargetEncoder` from `category_encoders` was employed to encode categorical features based on the target variable.
- **Handling Class Imbalance**: To address class imbalance, `SMOTE` was utilized to generate synthetic samples for the minority class.
- **Standardizing Numerical Features**: `StandardScaler` was applied to standardize all numerical features.

### 3. Model Training

- **Hyperparameter Optimization with Optuna**: `Optuna` was used to optimize hyperparameters for LightGBM by defining an objective function and running multiple trials to identify the best parameters.
- **LightGBM Model**: LightGBM was chosen for its efficiency and performance with large datasets. Cross-validation was performed using `StratifiedKFold` to ensure class proportions were preserved across splits.
- **Early Stopping**: Early stopping was implemented to prevent overfitting, while LightGBM's pruning callback facilitated trial pruning.

### 4. Model Evaluation

- **Out-of-Fold Predictions**: The model's performance was evaluated using out-of-fold predictions, resulting in an overall ROC-AUC score of **0.96025**.
- **Feature Importance Analysis**: Feature importance was analyzed to determine which features had the greatest influence on loan approval predictions.

### 5. Submission Preparation

- Following model training, predictions on the test set were averaged across folds to yield robust results.
- The submission file was generated with `loan_status` probabilities for each applicant.

## Important Functions and Parameters

- **Optuna Objective Function**: The objective function employed `StratifiedKFold` cross-validation to evaluate various combinations of hyperparameters suggested by Optuna. Parameters such as `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda` were optimized.

- **LightGBM Pruning Callback**: `LightGBMPruningCallback` was utilized to prune underperforming trials, thereby focusing computational resources on promising hyperparameter sets.

- **Callbacks**: LightGBM training employed callbacks for early stopping and log evaluation to ensure efficient convergence and model monitoring.

## Key Techniques

1. **Feature Engineering**: Generating new features significantly enhanced the model's ability to distinguish between approved and non-approved loans.
2. **Hyperparameter Optimization with Optuna**: Rigorous hyperparameter optimization allowed the model to reach its maximum potential by identifying the optimal parameter set.
3. **Addressing Class Imbalance**: The application of SMOTE to generate synthetic samples from the minority class mitigated bias toward the majority class.
4. **Cross-Validation**: Stratified K-Fold cross-validation ensured a robust evaluation by preserving the distribution of the target variable across all folds.

## Running the Code

- Place the dataset files (`train.csv`, `test.csv`, `sample_submission.csv`) in the appropriate directory.
- Execute the `loan_approval_prediction.ipynb` script using Google Colab or Jupyter Notebook.
- The final submission file (`loan_approval_submission_optimized.csv`) will be generated, containing the `loan_status` probabilities for each applicant in the test set.

## Conclusion

The presented solution achieved a high-performing model with a Kaggle ROC-AUC score of **0.96025**. The success of the model was attributed to the combination of feature engineering, hyperparameter optimization, and the effective handling of class imbalance.

For any questions or suggestions for further improvements, feel free to raise an issue in the repository or reach out.

## Discussion

Currently, the model ranks **488th out of 1548** participants in the competition. This standing reflects the efficacy of the implemented methods, but there remains room for improvement to achieve a higher ranking. Future improvements may include the exploration of additional feature engineering techniques, more advanced ensemble methods, and further hyperparameter tuning to push the model's performance closer to the top rankings. Moreover, leveraging domain knowledge and exploring additional data sources, if available, could further enhance predictive capabilities.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
