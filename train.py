import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


#Split data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.2 ,random_state= 42)

#Parameters for random forest

max_depth = 15
n_estimators = 15

#apply mlflow

with mlflow.start_run(): # Its to tell what all to log by mlflow
    rf = RandomForestClassifier(max_depth= max_depth , n_estimators= n_estimators)
    rf.fit (X_train,y_train)
    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    # This is logging code 
    mlflow.log_metric('accuracy',accuracy) 
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    print(accuracy)
    
