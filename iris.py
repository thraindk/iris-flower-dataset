# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:50:39 2022

@author: cedri
"""

# =============================================================================
# my machine learning exercise using the popular iris flower dataset
# https://archive.ics.uci.edu/ml/datasets/Iris
# 
# based on
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# by Jason Brownlee
# 
#     Installing the Python and SciPy platform.
#     Loading the dataset.
#     Summarizing the dataset.
#     Visualizing the dataset.
#     Evaluating some algorithms.
#     Making some predictions.
#
# =============================================================================

# 1 check prerequisites

# pip install scipy numpy matplotlib pandas sklearn seaborn


# 2 load libraries
import numpy as np
import seaborn as sns
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 3 load dataset

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = read_csv("iris.data", names=names)

# 4 inspect dataset (getting the basic idea)

print(df.shape) # rows, columns
print(df.columns) # column names
print(df.index) # index information
print(df.head()) # preview first 5 rows
print(df.describe()) # statistics

# class (column) distribution
print(df.groupby('class').size()) # each 33,33%

# 5 UNIVARIATE PLOTS (einzelne Variablen betrachten)

# extend with visualization

# box and whisker plots
# Whiskers enden beim Minimum/Maximum, spätestens aber bei 1.5 IQR
# Linien: 1. Quartil, Median, 3. Quartil (ggf. Ausreisser)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms -> overview of distribution
df.hist()
plt.show()


#ECDF = Verteilungsfunktion einer Stichprobe / die kumulierten relativen Häufigkeiten
# = Addiere Häufigkeiten beginnend bei der kleinsten Ausprägung (bis 1.0)

def my_ecdf(x): # Liste!
    # Tupel(Gr, rel. Häufigkeit) 
    rel = [(i, x.count(i) / len(x)) for i in sorted(set(x))]  # set unordered!
    sum=0;
    t=[]
    for i in rel:
        sum = sum + i[1] # rel. Häufigkeiten (Pos 1/1) Stück für Stück aufsummieren
        t.append((i[0], sum)) # kummulieren - hier mit append!
    return t

#Beispiel an sepal-width

ecdf_function = my_ecdf(list(df["sepal-width"].dropna().sort_values())) # Series zu Liste

# brauchen Werte aus Tupel - for loop!

x_value_m = [x[0] for x in ecdf_function] # x=sepal-width

y_value_m = [x[1] for x in ecdf_function] # y=Kum. Verteilung

plot = sns.scatterplot(x=x_value_m, y=y_value_m, color="blue")
plot.set(xlabel="sepal-width", ylabel="kummulative Verteilung")

#TODO plots aufhübschen, set?

# 6 MULTIVARIATE PLOTS (multiple variables, interaction)

# scatter plot matrix
scatter_matrix(df)
plt.show()

# correlations

df_clean=df.dropna()
df_clean=df_clean.drop("class", axis=1) # kategorial weg

# match each column and calculate the corrcoef

df_corr = df_clean.corr(method="pearson")
print(df_corr)


# let's have a closer look at the petal-width and petal-length variables (high coeff of 0.96)

sns.scatterplot(data=df,x="petal-length", y="petal-width", hue=df["class"]).set(xlabel="petal-length", ylabel="petal-width")

# iris-virginica has the tallest petals

plt.plot()

plt.show()

# 7 Create Models and test accuracy on unseen data -> using validation dataset


# Split-out validation dataset
array = df.values # df to array (col 4 = class)
X = array[:,0:4] # : excluding col 4!
y = array[:,4] # only col 4 (class)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1) # 80:20

# =============================================================================
# k-FOLD CROSS VALIDATION

# We will use stratified 10-fold cross validation to estimate model accuracy.
# 
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.
# 
# Stratified means that each fold or split of the dataset will aim to have the same distribution of example by class as exist in the whole training dataset.
# We are using the metric of ‘accuracy‘ to evaluate models.
# This is a ratio of the number of correctly predicted instances divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate).
# =============================================================================

# build model using 6 different algorithms

# =============================================================================
#     Logistic Regression (LR)
#     Linear Discriminant Analysis (LDA)
#     K-Nearest Neighbors (KNN).
#     Classification and Regression Trees (CART).
#     Gaussian Naive Bayes (NB).
#     Support Vector Machines (SVM).
# =============================================================================


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# evaluate each model in turn
results = []
names = []
print("accuracy of model")
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()

# svm seems to have the highest accuracy

# make predicitions on the validation dataset

model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# evaluate prediction results

print("accuracy score: ", accuracy_score(Y_validation, predictions))
print("confusion matrix:")
print(confusion_matrix(Y_validation, predictions))
print("classification report:")
print(classification_report(Y_validation, predictions))

# fin