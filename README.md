# Predicting House Prices with Machine Learning

### Overview

This project demonstrates how to predict house prices in California using various machine learning models, including Linear Regression, Random Forest, and XGBoost. The dataset used is the California Housing dataset from scikit-learn, which includes features like median income, house age, and location to predict median house values.

### Key Features

**Data Preprocessing:** Normalization using StandardScaler to ensure features are on the same scale.

**Custom Linear Regression:** Implementation of a simple linear regression model from scratch.

**Model Training:**

Linear Regression (custom implementation)

Random Forest Regressor

XGBoost Regressor

**Model Evaluation:** Performance metrics include RMSE (Root Mean Squared Error) and R² score.

**Feature Importance:** Visualization of feature importance using Random Forest.

**Model Persistence:** Saved trained models for future use.

### Results

| Model               | RMSE   | R² Score |
|---------------------|--------|----------|
| Linear Regression   | 0.746  | 0.576    |
| Random Forest       | 0.505  | 0.805    |
| XGBoost             | 0.472  | 0.830    |

**Best Model**: XGBoost achieved both:
- Highest R² score (0.830)
- Lowest RMSE (0.472)

This indicates superior predictive performance compared to the other models.

### Dependencies

Python 3.10+

**Libraries:**

numpy

pandas

scikit-learn

xgboost

matplotlib

joblib

**Install dependencies using:**

pip install numpy pandas scikit-learn xgboost matplotlib joblib

### Usage

**Data Loading:** The dataset is loaded directly from scikit-learn (fetch_california_housing).

**Preprocessing:** Features are scaled using StandardScaler.

**Model Training:**

Train and evaluate Linear Regression, Random Forest, and XGBoost.

Visualize feature importance.

### Future Improvements

**Hyperparameter Tuning:** Optimize model parameters using GridSearchCV or RandomizedSearchCV.

**Feature Engineering:** Explore additional features or transformations to improve accuracy.

**Deployment:** Create a web API (e.g., Flask) for real-time predictions.

### License

This project is open-source under the MIT License.
