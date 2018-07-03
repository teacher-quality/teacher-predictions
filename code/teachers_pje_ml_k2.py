# -*- coding: utf-8 -*-
"""
Created on Fri May 25 17:48:29 2018

@author: Franco
"""

import pandas as pd
import numpy as np
import os
%matplotlib inline
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.plotly as py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



df_path = os.getcwd() + '\data\entrancescore-evdocente-noid.csv'
df = None
#with open(df_path, encoding="utf-16") as f:
with open(df_path, "r") as f:
    df = pd.read_csv(f)

list(df)
df.head(3)


#Divide the sample distribution in 10 types:
df['xtile'] = pd.qcut(df.pf, 10, labels = ['percentil: '+str(i) for i in range(10)])
df['worst'] = (df['xtile'] == 'percentil: 0' )| (df['xtile'] == 'percentil: 1')| (df['xtile'] == 'percentil: 2')


#Create a histogram:
list(df)

#Generate graphs: Histograms
pyplot.hist((df.paaverbal.dropna()-df.paaverbal.mean())/df.paaverbal.std(), 100, alpha=0.5, label='PAA - Verbal')
pyplot.hist((df.pf.dropna()-df.pf.mean())/df.pf.std(), 100, alpha=0.5, label='PJ - Portfolio')
pyplot.legend(loc='upper right')
pyplot.show()

#Generate graphs: Scatters:
#NEM vs PF_PJE:
df1 = df.sort_values(by = ['pf_pje'])
df1['xtile'] = pd.qcut(df1.pf_pje, 50, labels = list(range(50)))
scatter1 = df1[['pf', 'paaverbal', 'paamat']].groupby(df1['xtile']).mean()
plt.scatter((scatter1.paaverbal.dropna()-scatter1.paaverbal.mean())/scatter1.paaverbal.std(), (scatter1.pf.dropna()-scatter1.pf.mean())/scatter1.pf.std(), c="r", alpha=0.5, label="Correlation")
plt.xlabel("PAA - Verbal")
plt.ylabel("PJ - Portfolio")
plt.legend(loc=2)
plt.show()

df['paaverbal'].isnull().sum()
df['paamat'].isnull().sum()



#Generate Machine Learning Models:
#Using regression trees:
from sklearn import datasets, linear_model, preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree, svm
from sklearn.metrics import accuracy_score, roc_auc_score

df = df.dropna()
X = df[['paaverbal','paamat']].dropna()
X['Average'] = (X['paaverbal'] + X['paamat'])/2
X_mean = np.mean(X,1)
X_transformed = (X- X.mean())/X.std()

Y = df['worst']

X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed[['paaverbal','paamat']], Y, test_size=0.2)


#Use logistic regression and make predictions and graphs:
#=======================================================


def plot_estimates(df_lr_test, variable, bins, npol, polfit):
    df_lr_test['xtile'] = pd.cut(df_lr_test.y_test_hat, bins, duplicates='drop', labels = list(range(bins)))
    scatter1 = df_lr_test.groupby(df_lr_test['xtile']).mean()
    scatter1.columns = ['y_test_hat','Verbal','Math', 'Average']
    if polfit == 1:
        z = np.polyfit(scatter1[variable], scatter1.y_test_hat, npol)
        f = np.poly1d(z)
        x_new = np.linspace(scatter1[variable].min(), scatter1[variable].max(), 100)
        y_new = f(x_new)
        plt.scatter(scatter1[variable], scatter1.y_test_hat, c="r", alpha=0.5, label="Correlation")
        plt.plot(x_new, y_new, c='gray')
        plt.xlabel("PAA - " + variable)
        plt.ylabel("PJ - Portfolio")
        plt.legend(loc=1)
        plt.show()
    else:
        plt.scatter(scatter1['variable'], scatter1.y_test_hat, c="g", alpha=0.5, label="Correlation")
        plt.xlabel("PAA - " + variable)
        plt.ylabel("PJ - Portfolio")
        plt.legend(loc=1)
        plt.show()



def gen_predictions(Classifier, X_train, X_test, y_train):
    from sklearn import tree, svm
    SVC = svm.SVC
    if Classifier == SVC:
        clf = Classifier(probability=True)
    else:
        clf = Classifier()
    clf = clf.fit(X_train, y_train)
    y_test_hat = clf.predict_proba(X_test)[:,1]  #Prob of bad Teacher

    ndf = y_test_hat.shape[0]
    df_ml_test = np.concatenate([y_test_hat.reshape(ndf,1), np.array(X_test['paaverbal']).reshape(ndf,1), np.array(X_test['paamat']).reshape(ndf,1)],1)
    df_ml_test = pd.DataFrame(data = df_ml_test, columns = ['y_test_hat','Verbal','Math'])
    df_ml_test['Average'] = (df_ml_test['Verbal'] + df_ml_test['Math'])/2
    df_ml_test = df_ml_test.sort_values(by = ['y_test_hat'])

    return df_ml_test, y_test_hat


# Options: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier svm.SVC


df_rf_test , y_test_hat = gen_predictions(RandomForestClassifier, X_train_transformed, X_test_transformed, y_train)

plot_estimates(df_rf_test,'Average', 35, 2, 1)
plot_estimates(df_rf_test,'Math', 35, 2, 1)
plot_estimates(df_rf_test,'Verbal', 35, 3, 1)









math_lb_list = []
lang_lb_list = []
avg_lb_list = []


for i in range(50):
    X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed, Y, test_size=0.2)

    clf = RandomForestClassifier()
    clf = clf.fit(X_train_transformed, y_train)
    y_test_hat = clf.predict_proba(X_test_transformed)[:,1]  #Prob of bad Teacher

    z = np.polyfit(y_test_hat, X_test_transformed['paaverbal'], 1) #Verbal
    f = np.poly1d(z)
    f = np.array(f)
    lowest_v_std = (.9-f[1])/f[0]
    lowest_v = lowest_v_std*X.std()[0] + X.mean()[0]

    z = np.polyfit(y_test_hat, X_test_transformed['paamat'], 1) #Matematicas
    f = np.poly1d(z)
    f = np.array(f)
    lowest_m_std = (.9-f[1])/f[0]
    lowest_m = lowest_m_std*X.std()[1] + X.mean()[1]

    z = np.polyfit(y_test_hat, X_test_transformed['Average'], 1) #Average
    f = np.poly1d(z)
    f = np.array(f)
    lowest_avg_std = (.9-f[1])/f[0]
    lowest_avg = lowest_avg_std*X.std()[2] + X.mean()[2]

    lang_lb_list.append(lowest_v)
    math_lb_list.append(lowest_m)
    avg_lb_list.append(lowest_avg)

pyplot.hist(np.array(lang_lb_list), 20, alpha=0.5, color='g', label='PAA - Verbal LB')
pyplot.hist(np.array(math_lb_list), 20, alpha=0.5, color='maroon',label='PAA - Math LB')
pyplot.hist(np.array(avg_lb_list), 20, alpha=0.5, color='blue', label='PAA - Average LB')
pyplot.legend(loc='upper right')
pyplot.show()




np.mean(np.array(lang_lb_list))
np.mean(np.array(math_lb_list))
np.mean(np.array(avg_lb_list))


#Now create histograms to see the ammount of people that could be out of the system because of lower bound:

pyplot.hist(X['Average'], 200, alpha=0.5,color = 'maroon', label='PAA - Average')
plt.axvline(x=np.mean(np.array(avg_lb_list)), color = 'red')
pyplot.legend(loc='upper right')
pyplot.show()





























#Now we have to find the optimal lower bound:
X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed, Y, test_size=0.2)

clf = RandomForestClassifier()
clf = clf.fit(X_train_transformed, y_train)
y_test_hat = clf.predict_proba(X_test_transformed)[:,1]  #Prob of bad Teacher


z = np.polyfit(y_test_hat, X_test_transformed['paaverbal'], 1) #Verbal
f = np.poly1d(z)
f = np.array(f)
lowest_score_std = (.9-f[1])/f[0]
lowest_score = lowest_score_std*X.std()[0] + X.mean()[0]

z = np.polyfit(y_test_hat, X_test_transformed['paamat'], 1) #Matematicas
f = np.poly1d(z)
f = np.array(f)
lowest_score_std = (.9-f[1])/f[0]
lowest_score = lowest_score_std*X.std()[1] + X.mean()[1]

z = np.polyfit(y_test_hat, np.mean(X_test_transformed,1), 1) #Average
f = np.poly1d(z)
f = np.array(f)
lowest_score_std = (.9-f[1])/f[0]
lowest_score = lowest_score_std*X_mean.std() + X_mean.mean()



#Let's start with random forests:
clf = svm.SVC(probability=True)
clf = clf.fit(X_train_transformed, y_train)
y_test_hat = clf.predict_proba(X_test_transformed)[:,1]  #Prob of bad Teacher
ndf = y_test_hat.shape[0]
df_lr_test = np.concatenate([y_test_hat.reshape(ndf,1), X_test_transformed[:,0].reshape(ndf,1), X_test_transformed[:,1].reshape(ndf,1)],1)
df_lr_test = pd.DataFrame(data = df_lr_test, columns = ['y_test_hat','Verbal','Math'])
df_lr_test = df_lr_test.sort_values(by = ['y_test_hat'])





#Heating Maps:

fig = plt.figure(figsize=(6,6))
plt.clf()

ax = fig.add_subplot(111)
toplot = INSERT MATRIX HERE
res = ax.imshow(toplot, cmap=plt.cm.viridis, vmin = 0)
cb = fig.colorbar(res,fraction=0.046, pad=0.04)


plt.title('Heatmap')

plt.xlabel('x-axis')
plt.ylabel('y-axis')

row = np.where(toplot == toplot.max())[0][0]
column= np.where(toplot == toplot.max())[1][0]

plt.plot(column,row,'*')

plt.savefig('plots/heatmap.png', format='png')
