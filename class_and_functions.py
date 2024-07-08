import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,classification_report,confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier  
import xgboost as xgb
from sklearn.model_selection import train_test_split,cross_val_score, GridSearchCV, KFold, RandomizedSearchCV,RepeatedStratifiedKFold
from sklearn.preprocessing import OneHotEncoder,StandardScaler,MinMaxScaler,LabelEncoder,Normalizer,MaxAbsScaler,FunctionTransformer
from sklearn.decomposition import PCA
import random
#from imblearn.combine import SMOTEENN
from feature_engine.encoding import CountFrequencyEncoder,OrdinalEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import pickle
from datetime import date
import datetime
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import PolynomialFeatures
from category_encoders import TargetEncoder
import os
import joblib
import mlflow
from mlflow.models.signature import infer_signature


# ------------------ CLASSES ---------------------

class CustomPipeline:
    """
    A custom pipeline class that wraps a transformer and provides fit, transform, and fit_transform methods.
    
    Parameters:
    transformer: object
        The transformer object to be wrapped by the pipeline.
    
    Methods:
    fit(X, y=None)
        Fit the pipeline on the input data.
        
        Parameters:
        X: array-like or sparse matrix of shape (n_samples, n_features)
            The input data.
        y: array-like of shape (n_samples,), default=None
            The target values.
        
        Returns:
        self: object
            Returns self.
    
    transform(X)
        Transform the input data using the pipeline.
        
        Parameters:
        X: array-like or sparse matrix of shape (n_samples, n_features)
            The input data.
        
        Returns:
        X_transformed: array-like or sparse matrix of shape (n_samples, n_features_transformed)
            The transformed data.
    
    fit_transform(X, y=None)
        Fit the pipeline on the input data and transform it.
        
        Parameters:
        X: array-like or sparse matrix of shape (n_samples, n_features)
            The input data.
        y: array-like of shape (n_samples,), default=None
            The target values.
        
        Returns:
        X_transformed: array-like or sparse matrix of shape (n_samples, n_features_transformed)
            The transformed data.
    
    save(folder_path, name)
        Save the pipeline as a .pkl file in the specified folder path.
        
        Parameters:
        folder_path: str
            The path to the folder where the pipeline should be saved.
        name: str
            The name of the pipeline file.
    """
    def __init__(self, transformer):
        if not isinstance(transformer, (TransformerMixin, BaseEstimator)):
            raise ValueError("The provided object is not a valid transformer.")
        self.pipeline = transformer
    
    def fit(self, X, y=None):
        try:
            self.pipeline.fit(X, y)
            return self
        except AttributeError as e:
            raise AttributeError("The provided transformer does not have a fit method.") from e

    def transform(self, X):
        try:
            return self.pipeline.transform(X)
        except AttributeError as e:
            raise AttributeError("The provided transformer does not have a transform method.") from e

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    @staticmethod
    def __count_pkl_files_in_folder(folder_path):
        """
        Count the number of .pkl files in the specified folder path.
        
        Parameters:
        folder_path: str
            The path to the folder.
        
        Returns:
        count: int
            The number of .pkl files in the folder.
        """
        files = os.listdir(folder_path)
        pkl_files = [f for f in files if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pkl')]
        return len(pkl_files)
    
    @staticmethod
    def __load(folder_path, name):
        """
        Load a pipeline from a .pkl file in the specified folder path.
        
        Parameters:
        folder_path: str
            The path to the folder.
        name: str
            The name of the pipeline file.
        
        Returns:
        pipeline: object
            The loaded pipeline object.
        
        Raises:
        IndexError: If the provided file does not exist.
        """
        try:
            file = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pkl') and f.split('_', 1)[1].replace('.pkl', '') == name][0]
            pipeline = joblib.load(os.path.join(folder_path, file))
            return pipeline
        except IndexError as e:
            raise IndexError(f"The provided file {name} does not exist.") from e
    
    def save(self, folder_path, name):
        """
        Save the pipeline as a .pkl file in the specified folder path.
        
        If a file with the same name (without the numeric prefix) already exists in the folder, it will be overwritten.
        
        Parameters:
        folder_path: str
            The path to the folder where the pipeline should be saved.
        name: str
            The name of the pipeline file.
        """
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            
            # Check if a file with the same name (without the numeric prefix) already exists
            existing_file = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pkl') and f.split('_', 1)[1].replace('.pkl', '') == name]
            
            if existing_file:
                # If it exists, load and overwrite that file
                pipeline = CustomPipeline.__load(folder_path, name )
                pipeline = self.pipeline
                print(f"Pipeline '{name}' loaded from '{os.path.join(folder_path, existing_file[0])}'")
                joblib.dump(pipeline, os.path.join(folder_path, existing_file[0]))
                print(f"The pipeline was overwritten as '{os.path.join(folder_path,  existing_file[0])}'")
            else:
                # If it doesn't exist, generate a new name for the file
                number_of_files = CustomPipeline.__count_pkl_files_in_folder(folder_path)
                name = str(number_of_files + 1) + "_" + name + '.pkl'
            
            # Save the pipeline as a .pkl file in the folder
                joblib.dump(self.pipeline, os.path.join(folder_path, name))
                print(f"Pipeline saved as '{os.path.join(folder_path, name)}'")
            
        except Exception as e:
            print(f"Error saving the pipeline: {e}")
            raise e

class OutlierTransformation(BaseEstimator, TransformerMixin):
    """
    A transformer class for handling outliers in a dataset.

    Parameters:
    -----------
    upper_bound : float, optional
        The upper bound quantile value used to replace outliers. Default is 0.999.
    lower_bound : float, optional
        The lower bound quantile value used to replace outliers. Default is 0.001.

    Attributes:
    -----------
    columns : list
        A list of column names in the dataset that are of type int64 or float64.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        self : OutlierTransformation
            The fitted transformer object.

    transform(X)
        Transform the dataset by replacing outliers with quantile values.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset.
    """
    def __init__(self, upper_bound=0.999, lower_bound=0.001):  
        self.columns = None
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        
    def fit(self, X, y=None):
        self.columns = [x for x in X.columns if X[x].dtype=='int64' or X[x].dtype=='float64']
        return self

    def transform(self, X):
        if self.columns is None:
            raise RuntimeError("The fit method must be called before transform.")
        for col in self.columns:
            X.loc[(X[col] <= np.quantile(X[col], self.lower_bound)), col] = np.quantile(X[col], self.lower_bound)
            X.loc[(X[col] >= np.quantile(X[col], self.upper_bound)), col] = np.quantile(X[col], self.upper_bound)
        return X
   
class Standarize(BaseEstimator, TransformerMixin):
    """
    A transformer that standardizes numerical columns in a DataFrame using StandardScaler.

    Attributes:
    -----------
    columns : list
        A list of column names in the dataset that are of type int64 or float64.
    scaler : StandardScaler
        The StandardScaler instance used for scaling the data.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : Standarize
            The fitted transformer object.

    transform(X)
        Transform the dataset by scaling numerical columns.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset. """
    def __init__(self):
        self.columns=None
        self.scaler=StandardScaler()
        
    def fit(self,X,y=None):
        self.columns=[x for x in X.columns if X[x].dtype=='int64' or X[x].dtype=='float64']
        self.scaler.fit(X[self.columns])
        return self

    def transform(self,X):
        if self.columns is None:
            raise RuntimeError("The fit method must be called before transform.")
        X[self.columns]=self.scaler.transform(X[self.columns])
        return X
    
class GenericFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    A transformer that generates new features by multiplying and dividing numerical columns.

    Attributes:
    -----------
    columns : list
        A list of column names in the dataset that are to be used for feature engineering.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : GenericFeatureEngineering
            The fitted transformer object.

    transform(X)
        Transform the dataset by generating new features.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset.
    """
    def __init__(self,columns):
        self.columns=columns
        
    def fit(self,X,y=None):
        for col in self.columns:
            if X[col].dtype not in ['int64','float64']:
                raise ValueError(f"Column {col} is not numeric")
        return self

    def transform(self,X):
        for i, col1 in enumerate(X[self.columns].columns):
            for j, col2 in enumerate(X[self.columns].columns):
                if i <= j:  # Evitar duplicados y la interacción de una columna consigo misma
                    # Multiplicación de columnas
                    new_col_mult = f"{col1}_x_{col2}"
                    X[new_col_mult] = X[col1] * X[col2]

                    # División de columnas (evitar división por cero)
                    new_col_div = f"{col1}/{col2}"
                    X[new_col_div] = X[col1] / (X[col2] + 1e-8)
        return X
    
class AddPolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer that adds polynomial features to the dataset.

    Parameters:
    -----------
    degree : int, optional
        The degree of the polynomial features. Default is 2.
    include_bias : bool, optional
        Whether to include a bias (intercept) column. Default is False.
    interaction_only : bool, optional
        Whether to include only interaction features. Default is False.
    order : str, optional
        Order of polynomial terms ('C' or 'F'). Default is 'C'.

    Attributes:
    -----------
    columns : list
        A list of column names in the dataset that are of type int64 or float64.
    poly : PolynomialFeatures
        The PolynomialFeatures instance used for generating polynomial features.
    feature_names : list
        A list of generated feature names.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : AddPolynomialFeatures
            The fitted transformer object.

    transform(X)
        Transform the dataset by adding polynomial features.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset with polynomial features.
    """
    def __init__(self,columns, degree=2, include_bias=False, interaction_only=False, order='C'):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.order = order
        self.poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias, interaction_only=interaction_only, order=order)
        self.columns = columns
        self.feature_names = None

    def fit(self, X, y=None):
        self.poly.fit(X[self.columns])
        self.feature_names = self.poly.get_feature_names_out(input_features=self.columns)
        return self

    def transform(self, X):
        if self.columns is None:
            raise RuntimeError("The fit method must be called before transform.")
        
        poly_features = self.poly.transform(X[self.columns])
        poly_df = pd.DataFrame(poly_features, columns=self.feature_names, index=X.index)
        
        # Drop the original columns that are being replaced by polynomial features
        X = X.drop(columns=self.columns)
        
        # Concatenate the polynomial features DataFrame with the original DataFrame
        X = pd.concat([X, poly_df], axis=1)
        return X
        
class ColumnCheck(BaseEstimator,TransformerMixin):
    """
    A transformer that ensures specified columns are of a given type.

    Parameters:
    -----------
    categorical_columns : list
        List of column names to check.

    Attributes:
    -----------
    categorical_columns : list
        List of column names to check.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : ColumnCheck
            The fitted transformer object.

    transform(X)
        Transform the dataset by ensuring specified columns are of the given type.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset with checked columns.
    """
    def __init__(self,categorical_columns=None):
        self.categorical_columns = categorical_columns

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        if self.categorical_columns is None:
            raise RuntimeError("The fit method must be called before transform.")
              
        for col in self.categorical_columns:
            X[col]=X[col].astype('category')
        return X
    
class OneHotEncoding(BaseEstimator, TransformerMixin):
    """
    A transformer that applies one-hot encoding to specified columns in the dataset.

    Parameters:
    -----------
    columns : list
        List of column names to be one-hot encoded.

    Attributes:
    -----------
    columns : list
        List of column names to be one-hot encoded.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : OnehotEncoding
            The fitted transformer object.

    transform(X)
        Transform the dataset by applying one-hot encoding to specified columns.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset with one-hot encoded columns.
    """
    def __init__(self,columns=None):
        self.columns = columns
    
    def fit(self, X, y=None):
        if self.columns==None:
            raise KeyError("Columns are none.")
        try:
            X[self.columns]
        except KeyError as e:
            raise KeyError("The provided column does not exist in the dataframe.") from e
        return self
    
    def transform(self, X):
        if self.columns is None:
            raise RuntimeError("The fit method must be called before transform.")
        X = pd.get_dummies(X, columns=self.columns)
        return X
    
class MinMaxScalerDF(BaseEstimator, TransformerMixin):
    """
    A transformer that scales numerical columns in a DataFrame using MinMaxScaler.

    Parameters:
    -----------
    feature_range : tuple, optional
        Desired range of transformed data. Default is (0, 1).

    Attributes:
    -----------
    columns : list
        A list of column names in the dataset that are of type int64 or float64.
    scaler : MinMaxScaler
        The MinMaxScaler instance used for scaling the data.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : MinMaxScalerDF
            The fitted transformer object.

    transform(X)
        Transform the dataset by scaling numerical columns.

        Parameters:
        X : pandas DataFrame
            The input dataset.

        Returns:
        X : pandas DataFrame
            The transformed dataset.
    """
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.scaler = MinMaxScaler(feature_range=self.feature_range)
        self.columns = None

    def fit(self, X, y=None):
        self.columns = [x for x in X.columns if X[x].dtype in ['int64', 'float64']]
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        if self.columns is None:
            raise RuntimeError("The fit method must be called before transform.")
        
        X_scaled = self.scaler.transform(X[self.columns])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.columns, index=X.index)
        
        X[self.columns] = X_scaled_df
        return X
    
class DropHighlyCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """
    A transformer that drops highly correlated features from a DataFrame based on a specified correlation threshold.

    Parameters:
    -----------
    threshold : float, optional
        The correlation threshold above which features are considered highly correlated and will be dropped. Default is 0.7.

    Attributes:
    -----------
    to_drop : list
        A list of column names that are highly correlated and will be dropped.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the dataset by identifying highly correlated features.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        self : DropHighlyCorrelatedFeatures
            The fitted transformer object.

    transform(X, y=None)
        Transform the dataset by dropping the highly correlated features.

        Parameters:
        X : pandas DataFrame
            The input dataset.
        y : None
            Ignored.

        Returns:
        X : pandas DataFrame
            The transformed dataset with highly correlated features dropped.
    """
    def __init__(self, threshold=0.7):
        self.threshold = threshold
        self.to_drop = []
    
    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.to_drop = [column for column in upper.columns if any(upper[column] > self.threshold)]
        return self
    
    def transform(self, X, y=None):
        return X.drop(self.to_drop, axis=1)
    
# ------------------ FUNCTIONS ---------------------
def load_transformers(folder_path):
    pkl_files = sorted(
        [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.endswith('.pkl')],
        key=lambda x: int(x.split('_')[0])
    )
    # Cargar los transformadores
    transformers = [joblib.load(os.path.join(folder_path, f)) for f in pkl_files]
    return transformers

def print_metrics(y_val, y_pred):
  precision, recall, thresholds = precision_recall_curve(y_val, y_pred)
  pr_auc = auc(recall, precision)
  accuracy = accuracy_score(y_val, y_pred)
  precision=precision_score(y_val,y_pred)
  recall=recall_score(y_val,y_pred)
  f1=f1_score(y_val,y_pred)
  auc_score = roc_auc_score(y_val, y_pred)
  report = classification_report(y_val, y_pred)
  print(f"Accuracy: {accuracy}")
  print(f"Precision: {precision}")
  print(f"Recall: {recall}")
  print(f"F1-Score: {f1}")
  print(f"AUC-Score: {auc_score}")
  print(f"PRAUC: {pr_auc:.4f}")

def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_true, y_pred),
                annot=True,
                fmt='g',
                xticklabels=['0', '1'],
                yticklabels=['0', '1'],
                ax=ax)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_xlabel('Prediction', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=17)
    fig.savefig('Visualizaciones/Confusion_matrix.png')
    return fig

def plot_roc_curve(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    fig.savefig('Visualizaciones/ROC_curve.png')
    return fig

def plot_feature_importances(clf, X, style="seaborn", plot_size=(10, 8)):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns[indices]
    fig, ax = plt.subplots(figsize=plot_size)
    ax.set_title("Feature Importances")
    ax.bar(range(len(indices)), importances[indices], color="b", align="center")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(feature_names, rotation=65)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    fig.savefig('Visualizaciones/Feature_importances.png')
    return fig

def plot_logistic_regression_feature_importances(clf, X, style="white", plot_size=(10, 8)):
    if style:
        sns.set_style(style)

    coefficients = clf.coef_[0]
    importances = np.abs(coefficients)
    indices = np.argsort(importances)[::-1]
    feature_names = X.columns[indices]

    fig, ax = plt.subplots(figsize=plot_size)
    ax.set_title("Logistic Regression Feature Importances")
    ax.bar(range(len(indices)), importances[indices], color="r", align="center")
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(feature_names, rotation=65)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Absolute Coefficient Value")

    fig.tight_layout()
    fig.savefig('Visualizaciones/Feature_importances.png')
    return fig

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    else:
        fig = plt.figure(figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    fig.savefig('Visualizaciones/Learning_curve.png')
    return fig

def Log(df):
    for i in ['Administrative', 'Informational', 'ProductRelated', 'TotalAdministrative', 'TotalInformational', 'TotalProductRelated', 'Total_Duration', 'PageValues', 'PageValues/Total_Duration', 'PageValues_x_Total_Duration','ExitRates','BounceRates','Total_BounceExitRates']:
        df[i] = np.log1p(df[f'{i}'])
    return df

def feature_engineering(df):
    for i in ['Administrative', 'Informational', 'ProductRelated']:
        df[f'Total{i}'] = df[f'{i}'] * df[f'{i}_Duration']
    df['Total_Duration'] = df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration']
    df['PageValues/Total_Duration'] = df.apply(lambda x: x['PageValues'] / x['Total_Duration'] if x['Total_Duration']!=0 else 0,axis=1)
    df['PageValues_x_Total_Duration'] = df['PageValues'] * df['Total_Duration']
    df['Total_BounceExitRates'] = df['BounceRates'] + df['ExitRates']
    df['HigherThanAverage']=df.apply(lambda x: 1 if x.VisitorType=='New_Visitor' and x.Total_Duration>df[df.VisitorType=='New_Visitor']['Total_Duration'].mean() else
                                     1 if x.VisitorType!='New_Visitor' and x.Total_Duration>df[df.VisitorType!='New_Visitor']['Total_Duration'].mean() else
                                     0,axis=1)
    return df