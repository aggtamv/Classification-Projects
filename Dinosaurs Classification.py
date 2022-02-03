# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 00:46:24 2021

@author: bakel
"""

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import  preprocessing
from sklearn import metrics

#Importing Dataa
data = pd.read_csv('Larson.csv', sep = ';', header = 0)
#Handling NA measurements
data.dropna(subset =['Clade'], inplace = True)
Ch = pd.to_numeric(data['CH'], errors = 'coerce')
Ch = pd.DataFrame(Ch, columns = ['CH'])
data['CH'] = Ch['CH']
df = data[['FABL', 'CH','BW', 'ADL','PDL', 'Clade']]
#Replacing 0 values with NAs
df = df.replace({0 : np.nan})
df.isna().sum()

#Imputation for missing values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(random_state=0, missing_values = np.nan)
imp.fit(df.drop(['Clade'], axis = 1))
IterativeImputer(random_state=0)
X = imp.transform(df.drop(['Clade'], axis = 1))
#Final dataset

#Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X = scaler.fit_transform(X)

#Encoding Genus labels
y = df['Clade']
encoder = preprocessing.LabelEncoder()
b = encoder.fit(y)
y = encoder.transform(y)

#Testing with raw data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Knn Classification
model = KNeighborsClassifier(n_neighbors = 9, p =2, metric = 'minkowski')
model.fit(x_train, y_train)
predicted = model.predict(x_test)
acc= model.score(x_test, y_test)
print(acc)
#Knc Classification
from sklearn.neighbors import NearestCentroid
knc = NearestCentroid()
knc.fit(x_train,y_train)
knc_pred = knc.predict(x_test)
acc_knc = knc.score(x_test, y_test)
print(acc_knc)

#PCA reduction
from sklearn.decomposition import PCA
pca = PCA(n_components = 5)
pca_plot = pca.fit_transform(X)
loadings = pd.DataFrame(pca.components_.T, columns = ['PC1', 'PC2', 'PC3','PC4', 'PC5'], index = df.drop(['Clade'], axis = 1).columns)
variance = np.cumsum(pca.explained_variance_ratio_)
num_pc = pca.n_features_
pc_list = ['PC'+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame(loadings, columns = ['PC1', 'PC2', 'PC3','PC4', 'PC5'])
loadings_df

#PCA and biplot plot
import matplotlib.pyplot as plt
import pylab as pl
def myplot(score, coeff, labels = None):
    xs = score[:, 0]
    ys = score[:, 1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c = y)
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color = 'r', alpha = 1)
        if labels is None:
            plt.text(coeff[i,0] * 1.15, coeff[i, 1] * 1.15, 'Var' + str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()
    
    
myplot(pca_plot[:,0:2],np.transpose(pca.components_[0:2, :]))
plt.show()

#LDA dimension reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver = 'svd')
train_data = lda.fit(pca_plot,y)
lda_pred = lda.predict(pca_plot)

#LDA prediction accuracy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y, lda_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y, lda_pred)))

X = lda.transform(pca_plot)

#Testing model for PCA + LDA
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Knn Classification
model = KNeighborsClassifier(n_neighbors = 9, p =2, metric = 'minkowski')
model.fit(x_train, y_train)
predicted = model.predict(x_test)
acc= model.score(x_test, y_test)
print(acc)
#Knc Classification
from sklearn.neighbors import NearestCentroid
knc = NearestCentroid()
knc.fit(x_train,y_train)
knc_pred = knc.predict(x_test)
acc_knc = knc.score(x_test, y_test)
print(acc_knc)

#Kernel PCA dimension reduction
from sklearn.decomposition import KernelPCA
kpca_model = KernelPCA(n_components = 5, kernel = 'rbf', gamma = 1)
X = kpca_model.fit_transform(X)

#KPCA plot
for i in range(0, X.shape[0]):
    if y[i] == 0:
        c1 = pl.scatter(X[i,0],X[i,1],c='r', marker='+')
    elif y[i] == 1:
            c2 = pl.scatter(X[i,0],X[i,1],c='g',    marker='o')
    elif y[i] == 2:
           c3 = pl.scatter(X[i,0],X[i,1],c='b',    marker='v')
    elif y[i] == 3:
           c4 = pl.scatter(X[i,0],X[i,1],c='m',    marker='s')
    elif y[i] == 4:
           c5 = pl.scatter(X[i,0],X[i,1],c='y',    marker='p')
pl.legend([c1, c2, c3, c4, c5], ['Aves', 'Dromaeosauridae', 'Paronychodon', 'Richardoestesia',
       'Troodontidae'])
pl.title('Dinosaurs Kpca plot')
pl.show()

#LDA dimension reduction
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(solver = 'svd')
train_data = lda.fit(X,y)
lda_pred = lda.predict(X)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y, lda_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y, lda_pred)))

X = lda.transform(X)

#Testing model for KPCA + LDA
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Knn Classification
model = KNeighborsClassifier(n_neighbors = 9, p =2, metric = 'minkowski')
model.fit(x_train, y_train)
predicted = model.predict(x_test)
acc= model.score(x_test, y_test)
print(acc)
#Knc Classification
from sklearn.neighbors import NearestCentroid
knc = NearestCentroid()
knc.fit(x_train,y_train)
knc_pred = knc.predict(x_test)
acc_knc = knc.score(x_test, y_test)
print(acc_knc)
    

#Tuning parameters for Knn Classifier
import sklearn
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
k_range = list(range(1, 15))
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn,param_grid,refit=True,verbose=2, cv = 10, scoring = 'accuracy')
grid.fit(x_train,y_train.ravel()) 
Best = grid.best_estimator_
print(grid.best_estimator_)

#Cross-Validation for used Classifiers
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train_sizes, train_scores, valid_scores = learning_curve(
    KNeighborsClassifier(n_neighbors = 9, p =2, metric = 'minkowski'), X, y, scoring = 'accuracy',
    train_sizes= [1, 300, 800, 1500, 2427], cv= ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0), n_jobs = -1)

train_scores_mean = np.mean(train_scores, axis =1)
Std_training = np.std(train_scores, axis = 1)

validation_scores_mean = np.mean(valid_scores, axis =1)
Std_test = np.std(valid_scores, axis = 1)

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, '--',label = 'Training score', color = 'b')
plt.plot(train_sizes, validation_scores_mean,label = 'Validation score', color = 'g')
plt.ylabel('Accuracy Score', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for Knn Classifier', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(top  = 0.90)
plt.ylim(bottom  = 0.70)

