import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

class DataCleanerFeatureSelector:
    def __init__(self):
        pass

    def remove_constant_variance(self, data):
        data = data.select_dtypes(include=[np.number])
        selector = VarianceThreshold(threshold=0)
        selector.fit(data)
        non_constant_columns = selector.get_support(indices=True)
        return data.iloc[:, non_constant_columns]

    def select_k_best_features(self, X, y, k=10):
        selector = SelectKBest(score_func=f_regression, k=k)
        selected_data = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return pd.DataFrame(selected_data, columns=selected_features)

    def recursive_feature_elimination(self, X, y, n_features_to_select=8):
        model = LinearRegression()
        selector = RFE(model, n_features_to_select=n_features_to_select, step=1)
        selector = selector.fit(X, y)
        selected_features = X.columns[selector.get_support(indices=True)]
        return X[selected_features]

    def tree_based_feature_selection(self, X, y, n_features=8):
        model = RandomForestRegressor(random_state=123)
        model.fit(X, y)
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        selected_features = X.columns[indices]
        return X[selected_features]

    def lasso_feature_selection(self, X, y, alpha=0.01):
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)
        selected_features = X.columns[lasso.coef_ != 0]
        return X[selected_features]

    def plot_correlation_heatmap(self, data):
        corr_matrix = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={'shrink': .5})
        plt.title('Correlation Heatmap')
        plt.show()

     # New methods for additional plots

    def plot_scatter(self, data, feature_x, feature_y):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=data[feature_x], y=data[feature_y])
        plt.title(f'Scatter Plot of {feature_x} vs {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.show()

    def plot_bar(self, data, feature):
        plt.figure(figsize=(8, 6))
        data[feature].value_counts().plot(kind='bar')
        plt.title(f'Bar Plot of {feature}')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.show()

    def plot_pie(self, data, feature):
        plt.figure(figsize=(8, 6))
        data[feature].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90)
        plt.title(f'Pie Chart of {feature}')
        plt.ylabel('')  # Remove the y-label for better appearance
        plt.show()

    def plot_physics_based_features(self, X, y, feature_name):
        # Assuming physics-based feature is related to 'Engine_Load' and some dynamics of the vehicle
        plt.figure(figsize=(10, 6))
        plt.plot(X[feature_name], y, 'o')
        plt.title(f'{feature_name} vs Target')
        plt.xlabel(feature_name)
        plt.ylabel('maintenance_flag')
        plt.show()

    def plot_multiple_heatmaps(self, X, selected_features):
        for feature in selected_features:
            plt.figure(figsize=(8, 6))
            sns.heatmap(X[[feature]].corr(), annot=True, cmap='coolwarm', square=True)
            plt.title(f'Heatmap of {feature}')
            plt.show() 

    def visualize_all_plots(self, data, X, y):
        for feature in X.columns:
            self.plot_scatter(data, feature, 'Maintenance_flag')
            self.plot_bar(data, 'Maintenance_flag')
            self.plot_pie(data, 'Region')
        for feature in ['Vibration', 'Engine_Load']:
            self.plot_physics_based_features(X, y, feature)
            self.plot_multiple_heatmaps(X, ['Engine_Load', 'Vibration', 'Vehicle_speed_sensor'])

data = pd.read_csv(r"C:\Users\Lenovo\OneDrive\Desktop\fleet_train.csv")
X = data.drop(columns=['Maintenance_flag'])
y = data['Maintenance_flag']

analyzer = DataCleanerFeatureSelector()
analyzer.visualize_all_plots(data, X, y)

    

    
