# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 11:17:35 2021

@author: bakel
"""

import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import  preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np



path = 'Muttard Carnivores2.csv'

#Loading data
Training_data = pd.read_csv(path, sep = ';', header = 0)
Training_data.set_index("MARK ID", inplace = True)
Training_data.rename(columns= {'Biomechanical Group' : 'Mech_Type'}, inplace = True)
Training_data.drop('VOLUME', axis = 1, inplace = True)
Training_data[['SURFACE','MAXIMUM DEPTH', 'MEAN DEPTH', 'MAXIMUM LENGTH','MAXIMUM WIDTH',
               'MAXIMUM DEPTH.1','AREA','WIDTH','ROUGHNESS','OPENING ANGLE','FLOOR RADIUS']].astype(float)
#Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

surface = Training_data['SURFACE'].values.reshape(-1, 1)
surface = scaler.fit_transform(surface)

maxdepth = Training_data['MAXIMUM DEPTH'].values.reshape(-1, 1)
maxdepth = scaler.fit_transform(maxdepth)

meandepth = Training_data['MEAN DEPTH'].values.reshape(-1, 1)
meandepth = scaler.fit_transform(meandepth)

maxlength = Training_data['MAXIMUM LENGTH'].values.reshape(-1, 1)
maxlength = scaler.fit_transform(maxlength)

maxwidth = Training_data['MAXIMUM WIDTH'].values.reshape(-1, 1)
maxwidth = scaler.fit_transform(maxwidth)

maxdepth1 = Training_data['MAXIMUM DEPTH.1'].values.reshape(-1, 1)
maxdepth1 = scaler.fit_transform(maxdepth1)

area = Training_data['AREA'].values.reshape(-1, 1)
area = scaler.fit_transform(area)

width = Training_data['WIDTH'].values.reshape(-1, 1)
width = scaler.fit_transform(width)

roughness = Training_data['ROUGHNESS'].values.reshape(-1, 1)
roughness = scaler.fit_transform(roughness)

openingAngle = Training_data['OPENING ANGLE'].values.reshape(-1, 1)
openingAngle = scaler.fit_transform(openingAngle)

floorRadius = Training_data['FLOOR RADIUS'].values.reshape(-1, 1)
floorRadius = scaler.fit_transform(floorRadius)


#Merging dataframe
df = np.concatenate((surface, maxdepth,meandepth, maxlength, maxwidth,
                     maxdepth1, area, width, roughness, openingAngle,floorRadius), axis = 1)

df = pd.DataFrame(df, columns = ['SURFACE', 'MAXIMUM DEPTH', 'MEAN DEPTH', 'MAXIMUM LENGTH', 'MAXIMUM WIDTH',
       'MAXIMUM DEPTH.1', 'AREA', 'WIDTH', 'ROUGHNESS', 'OPENING ANGLE', 
       'FLOOR RADIUS'])

X = df
y= Training_data['Mech_Type']
encoder = preprocessing.LabelEncoder()
b = encoder.fit(y)
y = encoder.transform(y)

#Feature selection
X = X.drop(['MAXIMUM DEPTH', 'MEAN DEPTH', 'MAXIMUM LENGTH', 'MAXIMUM WIDTH',
       'MAXIMUM DEPTH.1', 'WIDTH', 'OPENING ANGLE'], axis = 1)

#PCA construction
from sklearn.decomposition import PCA
pca = PCA(n_components = 11)
pca_plot = pca.fit_transform(X)
loadings = pd.DataFrame(pca.components_.T, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                      'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'], index = df.columns)
variance = np.cumsum(pca.explained_variance_ratio_)
num_pc = pca.n_features_
pc_list = ['PC'+str(i) for i in list(range(1, num_pc+1))]
loadings_df = pd.DataFrame(loadings, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5',
                                                      'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'])
loadings_df

#PCA plot
import pylab as pl
for i in range(0, pca_plot.shape[0]):
    if y[i] == 0:
        c1 = pl.scatter(pca_plot[i,0],pca_plot[i,1],c='r',    marker='+')
    elif y[i] == 1:
            c2 = pl.scatter(pca_plot[i,0],pca_plot[i,1],c='g',    marker='o')
pl.legend([c1, c2], ['Flesh Slicer', 'Bone Crunchers'])
pl.title('Muttard Carnivores dataset\n11 features')
pl.show()

#PCA-biplot plot
from pca import pca
model_pca = pca(n_components=2)
res = model_pca.fit_transform(X)
fig, ax = model_pca.scatter()
fig, ax = model_pca.biplot(n_feat = 4)


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#LDA dimension reduction after PCA
lda = LDA(solver = 'svd')
train_data = lda.fit(pca_plot,y)
lda_pred = lda.predict(pca_plot)
cm = confusion_matrix(y, lda_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y, lda_pred)))

X = lda.transform(pca_plot)

#Split into train and test set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)

#K Nearest Neighbor Classification
model = KNeighborsClassifier(n_neighbors = 11, p =2, metric = 'minkowski')
model.fit(x_train, y_train)
predicted = model.predict(x_test) 
acc= model.score(x_test, y_test)
print(acc)

#K Nearest Centroid Classification
from sklearn.neighbors import NearestCentroid
knc = NearestCentroid()
knc.fit(x_train,y_train)
knc_pred = knc.predict(x_test)
acc_knc = knc.score(x_test, y_test)
print(acc_knc)

#KPCA + LDA dimension reduction
#KPCA 
from sklearn.decomposition import KernelPCA
modelKpca = KernelPCA(n_components= 11, kernel = 'rbf', gamma = 10)
X = modelKpca.fit_transform(X)
#KPCA plot
import pylab as pl
for i in range(0, X.shape[0]):
    if y[i] == 0:
        c1 = pl.scatter(X[i,0],X[i,1], c='r',    marker='+')
    elif y[i] == 1:
            c2 = pl.scatter(X[i,0],X[i,1],c='g',    marker='o')
pl.legend([c1, c2], ['Flesh Slicers', 'Bone Crunchers'])
pl.title('Muttard Carnivores dataset\n4 features')
pl.show()

#LDA
lda = LDA(solver = 'svd')
train_data = lda.fit(X,y)
lda_pred = lda.predict(X)
cm = confusion_matrix(y, lda_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y, lda_pred)))

X = lda.transform(X)

#Testing models KPCA + LDA
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.2)
#Knn Classification
model = KNeighborsClassifier(n_neighbors = 11, p =2, metric = 'minkowski')
model.fit(x_train, y_train)
predicted = model.predict(x_test) 
acc= model.score(x_test, y_test)
print(acc)

#KNC Classification
knc = NearestCentroid()
knc.fit(x_train,y_train)
knc_pred = knc.predict(x_test)
acc_knc = knc.score(x_test, y_test)
print(acc_knc)



#Parameters tuning
import sklearn
from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
k_range = list(range(1, 15))
param_grid = dict(n_neighbors = k_range)
grid = GridSearchCV(knn,param_grid,refit=True,verbose=2, cv = 10, scoring = 'accuracy')
grid.fit(x_train,y_train.ravel()) 
Best = grid.best_estimator_
print(grid.best_estimator_)

#Cross-Validation for constructed models
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

train_sizes, train_scores, valid_scores = learning_curve(
    KNeighborsClassifier(n_neighbors = 11, p =2, metric = 'minkowski'), X, y, scoring = 'accuracy',
    train_sizes= [1, 20, 70, 100, 117], cv= ShuffleSplit(n_splits = 100, test_size = 0.2, random_state = 0), n_jobs = -1)

train_scores_mean = np.mean(train_scores, axis =1)
Std_training = np.std(train_scores, axis = 1)

validation_scores_mean = np.mean(valid_scores, axis =1)
Std_test = np.std(valid_scores, axis = 1)

plt.style.use('seaborn')
plt.plot(train_sizes, train_scores_mean, '--',label = 'Training score', color = 'b')
plt.plot(train_sizes, validation_scores_mean,label = 'Validation score', color = 'g')
plt.ylabel('Accuracy Score', fontsize = 14)
plt.xlabel('Training set size', fontsize = 14)
plt.title('Learning curves for KNN Classifier\nKPCA + LDA 4 features', fontsize = 18, y = 1.03)
plt.legend()
plt.ylim(top  = 0.85)
plt.ylim(bottom  = 0.4)