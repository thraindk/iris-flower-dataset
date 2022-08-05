# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:50:39 2022

@author: cedri
"""

# my machine learning exercise based on the popular iris flower dataset
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
# https://archive.ics.uci.edu/ml/datasets/Iris

# 1 check prerequisites

# pip install scipy numpy matplotlib pandas sklearn

def printPrerequisites():
    import sys
    print('Python: {}'.format(sys.version))
    # scipy
    import scipy
    print('scipy: {}'.format(scipy.__version__))
    # numpy
    import numpy
    print('numpy: {}'.format(numpy.__version__))
    # matplotlib
    import matplotlib
    print('matplotlib: {}'.format(matplotlib.__version__))
    # pandas
    import pandas
    print('pandas: {}'.format(pandas.__version__))
    # scikit-learn
    import sklearn
    print('sklearn: {}'.format(sklearn.__version__))
    
printPrerequisites()

# 2 load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
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

# 5 univariate Plots (einzelne Variablen betrachten)

# extend with visualization

# box and whisker plots
# Whiskers enden beim Minimum/Maximum, spätestens aber bei 1.5 IQR
# Linien: 1. Quartil, Median, 3. Quartil (ggf. Ausreisser)
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms -> overview of distribution
df.hist()
pyplot.show()


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

#TODO plots aufhübschen, set?

# 6 multivariate Plots (multiple variables, interaction)

# scatter plot matrix
scatter_matrix(df)
pyplot.show()

# 7 Create Models and test accuracy on unseen data -> using validation dataset