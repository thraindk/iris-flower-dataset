# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:50:39 2022

@author: cedri
"""

# =============================================================================
# my machine learning exercise based on the popular iris flower dataset
# https://archive.ics.uci.edu/ml/datasets/Iris
# 
# based on
# https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
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