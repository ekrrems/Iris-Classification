# Iris Flowers Classification Project

#1. Importing Iris Flowers Dataset

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


Irisds = sns.load_dataset('iris')
df =Irisds.copy()
df.head()

df.info()

# How many Species are in the dataset?
df['species'].value_counts()

#To view basic statistical details about dataset
df.describe().T

#Visualizing the Dataset variables and Correlation between them

sns.pairplot(df,hue='species') #Iris Setosa is seperated from both other species according to graphics

sns.heatmap(df.corr(),annot=True)

# *Model Building*

X = df.drop(['species'],axis=1)
y = df['species']

print('X Shape:{}\nY Shape:{}'.format(X.shape,y.shape))


# 1.Train Test Split

X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2,
                                                 random_state=42)

# 2. Model Creation

#Let’s test 6 different algorithms:
#   Logistic Regression (LR)
#   Linear Discriminant Analysis (LDA)
#   K-Nearest Neighbors (KNN).
#   Classification and Regression Trees (CART).
#   Gaussian Naive Bayes (NB).
#   Support Vector Machines (SVM)

models =[]

models.append(('LR',LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVC', SVC(gamma='auto')))

results =[]
model_names=[]

for name,model in models:
    cv_results = cross_val_score(model, X_train, y_train, scoring='accuracy')
    results.append(cv_results)
    model_names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    print('{}:{} and {}'.format(name,cv_results.mean(),cv_results.std()))


model = LinearDiscriminantAnalysis()
model.fit(X_train,y_train)

prediction = model.predict(X_test)

print('Test Accuracy:{}'.format(accuracy_score(y_test,prediction)))
print('Classification Report:{}'.format(classification_report(y_test,prediction)))

