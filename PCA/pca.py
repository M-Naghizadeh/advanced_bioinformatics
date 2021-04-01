#perform pca 

import pandas as pd;
import numpy as np;

mydata = pd.DataFrame({'Var1': np.random.rand(200), 
                        'Var2': np.random.rand(200),
                        'Var3': np.random.rand(200),
                        'Var4': np.random.rand(200),
                        'Var5': np.random.rand(200),
                        'Var6' : np.random.rand(200),
                        'Var_target': np.repeat(['A','B', 'C', 'D'], 50)})


mydata.head()


#standardize the data

from sklearn.preprocessing import StandardScaler

values = mydata.iloc[:, :6].values

values.shape


values_scaled = StandardScaler().fit_transform(values)


#Correlation matrix and plot

cov_df = np.corrcoef(values_scaled.T); cov_df

%matplotlib

import matplotlib.pyplot as plt
img = plt.matshow(cov_df, cmap = plt.cm.rainbow)
plt.colorbar(img, ticks = [-1, 0, 1], fraction = .045)
for x in range(cov_df.shape[0]):
    for y in range(cov_df.shape[1]):
        plt.text(x, y, "%0.3f" % cov_df[x, y], size = 12,
                color = 'black', ha = 'center', va = 'center')


plt.show()

#observe covariance matrix and calculate Eigenvalues and Eigenvectors

cov_mat = np.cov(values_scaled.T)

cov_mat

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

eig_vals

eig_vals/sum(eig_vals)

total = sum(eig_vals)

var_explained = [(i / total) for i in sorted(eig_vals, reverse = True)]

var_explained

cumulative_variance_explained = np.cumsum(var_explained)

cumulative_variance_explained

plt.bar(range(len(var_explained)), var_explained, alpha = .4)

plt.step(range(len(var_explained)), cumulative_variance_explained, color = 'red')

#Perform PCA 

from sklearn.decomposition import PCA

pca = PCA()

prcomp = pca.fit_transform(values_scaled)

prcomp.shape

prcomp[1:10]

prcomp_df = pd.DataFrame(data =prcomp)

prcomp_df.head()

plt.scatter(prcomp_df[0], prcomp_df[1])

colors = {'A': 'red', 'B': 'blue', 'C' : 'green', 'D': 'black'}

plt.scatter(prcomp_df[0],prcomp_df[1], c = mydata.Var_target.map(colors))


#Explained variance ratio with sklearn pca

Explained_variance = pca.explained_variance_ratio_;

Explained_variance

var_explained    #compare result from doing pca with sklearn and calculating eigenvalues with numpy


pd.Series(Explained_variance, index = ['pc' + str(i) for i in range(6)]).plot.bar()


##Example datasets to perform PCA on:

from sklearn import datasets

iris = datasets.load_iris()

iris.feature_names


df = pd.DataFrame(data = iris.data, columns = iris.feature_names)

df['Class'] = iris.target

df.head()


