import pandas as pd
import numpy as np
import os
%matplotlib inline
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.plotly as py
from scipy import stats
from sklearn import datasets, linear_model, preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree, svm
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression

#df_path = os.getcwd() + '\data\entrancescore-evdocente-noid.csv'
df_path = os.getcwd() + '\data\PAA-evdoc.csv'
df = None
#with open(df_path, encoding="utf-16") as f:
with open(df_path, "r") as f:
    df = pd.read_csv(f)

q_v    = df["paa_verbal"].quantile(0.99)
q_m = df["paa_matematica"].quantile(0.99)

df = df[(df.paa_verbal<q_v) | (df.paa_matematica<q_m)]

#Quick df clean:
scores = ['paa_verbal', 'paa_matematica', 'nem', 'gpa', 'pce_hria_y_geografia', 'pce_biologia', 'pce_cs_sociales', 'pce_fisica', 'pce_matematica', 'pce_quimica']
cat = ['region', 'male']
df_s = df[scores].copy()
x_mean = df[scores].mean()
x_std = df[scores].std()
df_s = (df[scores] - df[scores].mean())/df[scores].std()
df_s['Average'] = (df['paa_verbal'] + df['paa_matematica'])/2

df_c = df[cat].copy()

df_s['took_hist'] = (df.pce_hria_y_geografia == 0)
df_s['took_bio'] = (df.pce_biologia == 0)
df_s['took_soc'] = (df.pce_cs_sociales == 0)
df_s['took_fis'] = (df.pce_fisica == 0)
df_s['took_mat'] = (df.pce_matematica == 0)
df_s['took_qui'] = (df.pce_quimica == 0)

y = df['pf_pje']

df = pd.concat([df_s, df_c, y], axis=1, sort=False)

df = df.dropna()
df.region = df.region.astype(int)
df.male = df.male.astype(int)

#Generate dichotomous variables by region
for x in set(df.region):
    df['region' + str(x)] = df.region == x

#Divide the sample distribution in 10 types:
df['xtile'] = pd.qcut(df.pf_pje, 10, labels = ['percentil: '+str(i) for i in range(10)])
df['worst'] = (df['xtile'] == 'percentil: 0' )| (df['xtile'] == 'percentil: 1')


df.columns = ['paaverbal' if x=='paa_verbal' else 'paamat' if x =='paa_matematica' else x for x in df.columns] #Change colname

x_variables = [x if (x!='pf_pje' and x!='xtile' and x!='worst' and x!='region9') else "AAA" for x in list(df)]
x_variables = sorted(list(set(x_variables)))
del x_variables[0]

X_transformed = df[x_variables].dropna().drop(columns = ['region', 'pce_biologia', 'pce_cs_sociales', 'pce_fisica', 'pce_hria_y_geografia', 'pce_matematica', 'pce_quimica',
'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7', 'region8', 'region10', 'region11', 'region12', 'region13', 'region14', 'region15',
'took_bio', 'took_fis', 'took_hist', 'took_mat', 'took_qui', 'took_soc'])
Y = df['worst']
X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed, Y, test_size=0.15)

resampling = 0
if resampling == 1:
    Train_set = X_train_transformed.copy()
    Test_set = X_test_transformed.copy()
    Train_set['y_train'] = y_train
    Train_set['random'] = np.random.randn(X_train_transformed.shape[0],1)
    n_true = Train_set[Train_set.y_train == 1].shape[0]
    n_false = Train_set[Train_set.y_train == 0].shape[0]
    n_start = n_false - n_true
    Train_set = Train_set.sort_values(by = ['y_train','random']).reset_index().iloc[n_start:]

    x_variables = [x if (x!='index' and x!='random' and x!='y_train' and x!='region9' ) else 'AAA' for x in list(Train_set)]
    x_variables = sorted(list(set(x_variables)))
    del x_variables[0]
    X_train_transformed = Train_set[x_variables]
    X_test_transformed = Test_set[x_variables]
    y_train = Train_set.y_train

'''
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
    df['paamat'].isnull().sum()`
'''



#Define some useful functions:
#============================

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
        plt.ylabel("Prob - Bad Teacher")
        plt.legend(loc=1)
        plt.show()
    else:
        plt.scatter(scatter1['variable'], scatter1.y_test_hat, c="g", alpha=0.5, label="Correlation")
        plt.xlabel("PAA - " + variable)
        plt.ylabel("Prob - Bad Teacher")
        plt.legend(loc=1)
        plt.show()


def gen_predictions(Classifier, X_train, X_test, y_train):
    from sklearn import tree, svm
    SVC = svm.SVC
    if Classifier == SVC:
        clf = Classifier(C=1, probability=True)
    else:
        clf = Classifier()
    clf = clf.fit(X_train, y_train)

    y_test_hat = clf.predict_proba(X_test)[:,1]  #Prob of bad Teacher
    y_test_predict = clf.predict(X_test)  #Prob of bad Teacher

    ndf = y_test_hat.shape[0]
    df_ml_test = np.concatenate([y_test_hat.reshape(ndf,1), np.array(X_test['paaverbal']).reshape(ndf,1), np.array(X_test['paamat']).reshape(ndf,1)],1)
    df_ml_test = pd.DataFrame(data = df_ml_test, columns = ['y_test_hat','Verbal','Math'])
    df_ml_test['Average'] = (df_ml_test['Verbal'] + df_ml_test['Math'])/2
    df_ml_test = df_ml_test.sort_values(by = ['y_test_hat'])

    return df_ml_test, clf.predict_proba(X_test), y_test_predict

# Options: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier svm.SVC
df_rf_test , y_test_hat, y_test_predict = gen_predictions(RandomForestClassifier, X_train_transformed.drop(columns = ['Average']), X_test_transformed.drop(columns = ['Average']), y_train)

#Accuracy tests:
accuracy = np.mean(y_test_predict == y_test)
type1_error = 1 - np.mean(y_test[y_test==1] == y_test_predict[y_test==1]) #Type 1 error: Probabilidad de predecir que no es malo cuando si lo es
type2_error = 1 - np.mean(y_test[y_test_predict == 1] == y_test_predict[y_test_predict == 1]) #probabilidad de predecir que profesor es malo cuando en realidad es bueno (Error tipo 2)

#Plot results
plot_estimates(df_rf_test,'Average', 15, 1, 1)
plot_estimates(df_rf_test,'Math', 15, 1, 1)
plot_estimates(df_rf_test,'Verbal', 15, 1, 1)

plt.scatter(y_test_hat[:,1],df_rf_test.Verbal)

# Sensibility analysis:
accuracy_list = []
pr_list = list(np.unique(y_test_hat[:,1][y_test_hat[:,1]>=.5]))
for pr in pr_list:
    acc = np.mean(y_test[y_test_hat[:,1]>=pr] == y_test_predict[y_test_hat[:,1]>=pr])
    accuracy_list.append(acc)

ticks = np.arange(len(accuracy_list))
ticks_label = [str(x) for x in pr_list]

plt.bar(ticks,accuracy_list, align='center', alpha=0.5, color = 'g')
plt.xticks(ticks, ticks_label)
plt.xlabel('Y hat')
plt.ylabel('Accuracy: 1 - Pr(failure)')
plt.title('Likelihood of failure when predicting bad teacher')
plt.show()

#Then: we should work with RandomForestClassifier

##########################################
#Start with the Score Sensibility analysis
##########################################

#CrossValidation Excercise Accuracy tests:
accuracy_type2 = []
accuracy_overall = []
for i in range(10):

    X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed.drop(columns = ['Average']), Y, test_size=0.15)
    clf = RandomForestClassifier()
    clf = clf.fit(X_train_transformed, y_train)
    y_test_hat = clf.predict(X_test_transformed)  #Prob of bad Teacher
    acc_overall = np.mean(y_test == y_test_hat)
    acc_type2 = np.mean(y_test[y_test_hat==1] == y_test_hat[y_test_hat==1])
    accuracy_overall.append(acc_overall)
    accuracy_type2.append(acc_type2)

plt.hist(accuracy_overall, alpha=0.3, color = 'g', label = 'Overall')
plt.hist(accuracy_type2, alpha=0.3, color = 'maroon', label = 'Y = 1')
plt.xlabel('Prediction success')
plt.ylabel('Density')
plt.title('Cross validation: Accuracy of Random Forest')
plt.legend(loc=1)
plt.show()





#Validate score Sensibility Analysis

math_lb_list = []
lang_lb_list = []
avg_lb_list = []

lang_mean = x_mean[0]
mat_mean = x_mean[1]
avg_mean = (lang_mean + mat_mean)/2

lang_std = x_std[0]
mat_std = x_std[1]
avg_std = x_std[2]


avg_std_list = [(300 + 3*i)  for i in range(101)]
avg_std_list = (avg_std_list - avg_mean)/avg_std

plt.scatter(X_test_transformed['paamat'],y_test_hat)

for i in range(20):

    X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed.drop(columns = ['Average']), Y, test_size=0.15)
    clf = RandomForestClassifier()
    clf = clf.fit(X_train_transformed, y_train)
    y_test_hat = clf.predict_proba(X_test_transformed)[:,1]  #Prob of bad Teacher

    clf.classes_

    z = np.polyfit(y_test_hat, X_test_transformed['paamat'], 1) #Verbal
    f = np.poly1d(z)
    f = np.array(f)

    lowest_m_std = (.25-f[1])/f[0]
    lowest_m = lowest_v_std*mat_std + mat_mean


    z = np.polyfit(y_test_hat, X_test_transformed['gpa'], 1) #Verbal
    f = np.poly1d(z)
    f = np.array(f)

    lowest_v_std = (.50-f[1])/f[0]
    lowest_v = lowest_v_std*lang_std + lang_mean



    z = np.polyfit(scatter1[variable], scatter1.y_test_hat, npol)
    f = np.poly1d(z)
    x_new = np.linspace(scatter1[variable].min(), scatter1[variable].max(), 100)
    y_new = f(x_new)
    y_pred_list = avg_std_list*f[0]  + f[1]

    for pred in y_pred_list:
        y_hat = np.mean(y_test[y_test_hat> 0.6])


    if i % 500 == 0:
        print('Iteration number: ' + str(i))




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




pyplot.hist(np.array(lang_lb_list), 50, alpha=0.5, color='g', label='PAA - Verbal LB')
pyplot.hist(np.array(math_lb_list), 50, alpha=0.5, color='maroon',label='PAA - Math LB')
pyplot.hist(np.array(avg_lb_list), 50, alpha=0.5, color='blue', label='PAA - Average LB')
pyplot.legend(loc='upper right')
pyplot.show()

np.mean(np.array(lang_lb_list))
np.mean(np.array(math_lb_list))
np.mean(np.array(avg_lb_list))


#Now create histograms to see the ammount of people that could be out of the system because of lower bound:

pyplot.hist(X['Average'], 200, alpha=0.5, label='PAA - Average')
plt.axvline(x=np.mean(np.array(avg_lb_list)))
pyplot.legend(loc='upper right')
pyplot.show()

np.mean(X['Average'] <= np.mean(np.array(avg_lb_list)))




np.mean(y_test[y_test_hat> 0.5])

np.mean(y_test[y_test_hat> 0.7])
np.mean(y_test[y_test_hat> 0.8])
np.mean(y_test[y_test_hat> 0.9])














scatter = X_train_transformed.copy()
scatter['y'] = y_train
scatter['randn'] = np.random.randn(X_train_transformed.shape[0])
scatter = scatter.sort_values(by = ['randn'])
scatter = scatter[scatter['randn'] < -2]
scatter.shape
scatter.randn.mean()



plt.scatter(scatter.paaverbal[y_train == 1], scatter.paamat[y_train == 1], c="r", alpha=0.5, label="Bad Teachers")
plt.scatter(scatter.paaverbal[y_train == 0], scatter.paamat[y_train == 0], c="b", alpha=0.5, label="Good Teachers")
plt.xlabel("PAA - Verbal")
plt.ylabel("PAA - Math")
plt.legend(loc=2)
plt.show()






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
