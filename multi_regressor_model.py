from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

class MultiRegressorModel:
    def __init__(self):
        # Define models
        self.models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'ElasticNet': ElasticNet(),
            'DecisionTree': DecisionTreeRegressor(),
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'XGBoost': XGBRegressor(),
            'KNN': KNeighborsRegressor(),
            'SVR': SVR()
        }
        self.best_models = {}

    def tune_and_train_models(self, X_train, y_train, param_grids):
        for name, model in self.models.items():
            print(f"Training {name}...")
            grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5, scoring='r2', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            self.best_models[name] = grid_search.best_estimator_

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.best_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                "R2 Score": model.score(X_test, y_test),
                "Mean Squared Error": mean_squared_error(y_test, y_pred),
                "Mean Absolute Error": mean_absolute_error(y_test, y_pred)
            }
            print(f"{name} - R2 Score: {results[name]['R2 Score']}, MSE: {results[name]['Mean Squared Error']}, MAE: {results[name]['Mean Absolute Error']}")
        return results
