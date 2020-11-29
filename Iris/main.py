import pandas as pd
import numpy as np

from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer

from sklearn.preprocessing import LabelEncoder

#________________________________________________________________________________________________________________

iris_data = pd.read_csv("iris.csv", engine="python")    #reads in data
print(iris_data.head())                                 #prints first 5 rows of data

iris_data.info()                                        #gives info about data

print(iris_data.shape[0])                               #number of rows in data
print(iris_data.shape[1])                               #number of columns in data
print(iris_data.shape)                                  #rows, cols

print(iris_data.variety.unique())                       #prints unique varieties of iris in the dataset

print(iris_data.tail())                                 #prints last 5 rows of data

print(iris_data.isnull())                               #prints out false for each row x col if it is not null

print(iris_data.describe())                             #prints out count, mean, std. quartiles, etc... for cols in data

def missing_values(x):                                  #defining function to test for missing values in dataset
    return sum(x.isnull())

print("Missing values in each column:")
print(iris_data.apply(missing_values,axis=0))           #printing out missing values in each col of dataset



