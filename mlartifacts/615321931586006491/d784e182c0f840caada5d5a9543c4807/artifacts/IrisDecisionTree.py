#Iris dataset with Decision Tree classifier for new exeriment in MLflowimport mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

mlflow.set_tracking_uri('http://localhost:5000') # Setting up mlflow tracking server locally
#Load Iris dataset
iris = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
X = iris.iloc[:,0:-1]
y = iris.iloc[:,-1]


#Split data
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=.2 ,random_state= 42)

#Parameters for random forest

max_depth = 15


#apply mlflow

mlflow.set_experiment('IrisDecisionTree') # naming experiment

with mlflow.start_run(): # Its to tell what all to log by mlflow
    irisdt = DecisionTreeClassifier(max_depth= max_depth)
    irisdt.fit (X_train,y_train)
    y_pred = irisdt.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    # This is logging code 
    mlflow.log_metric('accuracy',accuracy) #log metric
    mlflow.log_param('max_depth',max_depth) # log parameters
    # You can track code , model , artifacts as well
   
   #Create coonfusion matrix
    cm= confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues' )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matric')

    plt.savefig('Confusion_matrix.png') # Save plot as artifact
    mlflow.log_artifact("Confusion_matrix.png") # log artifacts with path to file
    mlflow.log_artifact(__file__) # log this code file as well
    mlflow.sklearn.log_model(irisdt,'Iris Decision Tree') # log model .
    mlflow.set_tag('CreatedBY','Shishant') # Tags for searching when lot of runs in place
    mlflow.set_tag('AlgoUsed','DecisionTree')

    #Log input dataset

    traindf=X_train
    traindf['variety']=y_train

    testdf=X_test
    testdf['variety']=y_test

    traindf = mlflow.data.from_pandas(traindf)
    testdf = mlflow.data.from_pandas(testdf)

    mlflow.log_input(traindf,context='train dataset')
    mlflow.log_input(testdf,context='test dataset')

    print(accuracy)    
