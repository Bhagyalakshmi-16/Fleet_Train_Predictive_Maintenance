import pandas as pd
from sklearn.model_selection import train_test_split
from multi_classifier_model import MultiClassifierModel
from multi_regressor_model import MultiRegressorModel
from data_cleaner_feature_selector import DataCleanerFeatureSelector

def main():
    # Load your dataset
    df = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\fleet_train.csv")

    # Split data into features and target
    X = df.drop('Maintenance_flag', axis=1)
    y = df['Maintenance_flag']

    # Initialize the DataCleanerFeatureSelector class
    data_cleaner = DataCleanerFeatureSelector()

    # Data cleaning and feature selection
    X_clean = data_cleaner.remove_constant_variance(X)
    X_selected = data_cleaner.select_k_best_features(X_clean, y, k=10)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Initialize the MultiClassifierModel class
    classifier_trainer = MultiClassifierModel()

    # Define hyperparameters for each model
    clf_param_grids = {
    'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
    'KNN': {'n_neighbors': [3, 5, 7]},
    'LogisticRegression': {'C': [0.1, 1.0, 10]},
    'DecisionTree': {'max_depth': [None, 10, 20]},
    'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
    'MLPClassifier': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']},
    'NaiveBayes': {}   # No hyperparameters for Naive Bayes
    }
    # Train and tune classification models
    classifier_trainer.tune_and_train_models(X_train, y_train, clf_param_grids)

    # Evaluate classification models on the test set
    clf_results = classifier_trainer.evaluate_models(X_test, y_test)

    # Print classification results
    print("Classification Results:", clf_results)

    
    # Initialize the MultiRegressorModel class
    regressor_trainer = MultiRegressorModel()

    # Define hyperparameters for each regression model
    reg_param_grids = {
        'LinearRegression': {},
        'Ridge': {'alpha': [0.1, 1.0, 10]},
        'Lasso': {'alpha': [0.1, 1.0, 10]},
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [None, 10, 20]},
        'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
        'SVR': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
        'KNN': {'n_neighbors': [3, 5, 7]},
        'DecisionTree': {'max_depth': [None, 10, 20]},
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']}
    }    
    # Train and tune regression models
    regressor_trainer.tune_and_train_models(X_train, y_train, reg_param_grids)

    # Evaluate regression models on the test set
    reg_results = regressor_trainer.evaluate_models(X_test, y_test)

    # Print  regression results
    print("Regression Results:", reg_results)


if __name__ == "__main__":
    main()
