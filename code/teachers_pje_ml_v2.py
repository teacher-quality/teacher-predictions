import pandas as pd
import numpy as np
import os
import seaborn as sns
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


q_v    = df["paa_verbal"].quantile(0.95)
q_m = df["paa_matematica"].quantile(0.95)

df = df[(df.paa_verbal<q_v) | (df.paa_matematica<q_m)]
#Quick df clean:
scores = ['paa_verbal', 'paa_matematica', 'nem', 'gpa', 'pce_hria_y_geografia', 'pce_biologia', 'pce_cs_sociales', 'pce_fisica', 'pce_matematica', 'pce_quimica']
cat = ['region', 'male']
df_s = df[scores].copy()

bygroup = 0
if bygroup == 1:
    df_s['periodo'] = df['periodo']
    df_s = df_s.groupby('periodo').transform(lambda x: (x - x.mean()) / x.std())
else:
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
df['worst'] = (df['xtile'] == 'percentil: 0' ) | (df['xtile'] == 'percentil: 1') | (df['xtile'] == 'percentil: 2')

df.columns = ['paaverbal' if x=='paa_verbal' else 'paamat' if x =='paa_matematica' else x for x in df.columns] #Change colname

x_variables = [x if (x!='pf_pje' and x!='xtile' and x!='worst' and x!='region9') else "AAA" for x in list(df)]
x_variables = sorted(list(set(x_variables)))
del x_variables[0]

X_transformed = df[x_variables].dropna().drop(columns = ['region',
'pce_biologia', 'pce_cs_sociales', 'pce_fisica', 'pce_hria_y_geografia', 'pce_matematica', 'pce_quimica',
'region1', 'region2', 'region3', 'region4', 'region5', 'region6', 'region7', 'region8', 'region10', 'region11', 'region12', 'region13', 'region14', 'region15'])
#'took_bio', 'took_fis', 'took_hist', 'took_mat', 'took_qui', 'took_soc', 'paamat', 'paaverbal'

X_transformed['paa_avg'] = (X_transformed.Average - X_transformed.Average.mean())/X_transformed.Average.std()

Y = df['worst']

X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed.drop(columns = ['Average', 'paa_avg', 'gpa']), Y, test_size=0.15)

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


import os
os.environ["PATH"] += os.pathsep + 'C:\\Users\\Franco\\Anaconda3\\Library\\bin\\graphviz'
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_leaf_nodes = 8)
dtree.fit(X_train_transformed,y_train)
#dtree.fit(X_train_transformed.drop(columns = ['paamat', 'paaverbal', 'male']),y_train)

dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())



#Define some useful functions:
#============================

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
    df_ml_test = X_test.copy()
    df_ml_test['y_test_hat'] = y_test_hat
    df_ml_test = df_ml_test.sort_values(by = ['y_test_hat'])

    return df_ml_test, clf.predict_proba(X_test), y_test_predict


# Options: RandomForestClassifier, LogisticRegression, DecisionTreeClassifier svm.SVC
df_rf_test , y_test_hat, y_test_predict = gen_predictions(RandomForestClassifier, X_train_transformed, X_test_transformed, y_train)
#df_rf_test.to_csv(r'C:\Users\Franco\GitHub\teacher-predictions\output\data_predictions.csv')

#Accuracy tests:
accuracy = np.mean(y_test_predict == y_test)
type1_error = 1 - np.mean(y_train[y_train==1] == y_test_predict[y_train==1]) #Type 1 error: Probabilidad de predecir que no es malo cuando si lo es
type2_error = 1 - np.mean(y_train[y_test_predict == 1] == y_test_predict[y_test_predict == 1]) #probabilidad de predecir que profesor es malo cuando en realidad es bueno (Error tipo 2)

#Plot results

sns.regplot(df_rf_test.paamat, df_rf_test.y_test_hat, x_bins = 15)
plt.ylabel(r'$\^Yprob$')
plt.xlabel('PAA Math')
plt.savefig(r'C:\\Users\\Franco\\GitHub\\teacher-predictions\\output\\Pr_vs_paa_mat.pdf', bbox_inches='tight')

sns.regplot(df_rf_test.paaverbal, df_rf_test.y_test_hat, x_bins = 15)
plt.ylabel(r'$\^Yprob$')
plt.xlabel('PAA Verbal')
plt.savefig(r'C:\\Users\\Franco\\GitHub\\teacher-predictions\\output\\Pr_vs_paa_verbal.pdf', bbox_inches='tight')

df_rf_test['paa_avg'] = (df_rf_test.paaverbal + df_rf_test.paamat)/2
sns.regplot(df_rf_test.paa_avg, df_rf_test.y_test_hat, x_bins = 15)
plt.ylabel(r'$\^Yprob$')
plt.xlabel('PAA Average')
plt.savefig(r'C:\\Users\\Franco\\GitHub\\teacher-predictions\\output\\Pr_vs_paa_avg.pdf', bbox_inches='tight')


sns.regplot(df_rf_test.nem, df_rf_test.y_test_hat, x_bins = 15)
plt.ylabel(r'$\^Yprob$')
plt.xlabel('GPA')
plt.savefig(r'C:\\Users\\Franco\\GitHub\\teacher-predictions\\output\\Pr_vs_gpa.pdf', bbox_inches='tight')



# Sensibility analysis:
accuracy_list = []
pr_list = list(np.unique(y_test_hat[:,1][y_test_hat[:,1]>=.5]))
for pr in pr_list:
    acc = np.mean(y_test[y_test_hat[:,1]>=pr] == y_test_predict[y_test_hat[:,1]>=pr])
    accuracy_list.append(acc)

ticks = np.arange(len(accuracy_list))
pr_list = [round(x, 2) for x in pr_list]
ticks_label = [str(x) for x in pr_list]

plt.bar(ticks,accuracy_list, align='center', alpha=0.7, color = 'gray')
plt.xticks(ticks, ticks_label)
plt.xlabel('Y hat')
plt.ylabel('Accuracy: 1 - Pr(failure)')
plt.savefig(r'D:\\Google Drive\\Teacher Quality\\4. Work\\Prediction\\Appendix\\accuracy.pdf', bbox_inches='tight')





import numpy as np
import pylab as plt

X_train_transformed.head(1)
X_train2 = X_train_transformed.drop(columns = ['male', 'took_bio', 'took_fis', 'took_hist', 'took_mat', 'took_qui', 'took_soc', 'nem'])

clf = RandomForestClassifier(max_leaf_nodes=8)
#clf = RandomForestClassifier()
clf = clf.fit(X_train2, y_train)
y_test_hat = clf.predict_proba(X_train2)[:,1]  #Prob of bad Teacher
X_train2.head(1)
# Sample data
grid_size = 100
mat_vals = np.linspace(X_train2['paamat'].min(), X_train2['paamat'].max(), grid_size)
len_vals = np.linspace(X_train2['paaverbal'].min(), X_train2['paaverbal'].max(), grid_size)

R = np.empty((grid_size, grid_size))

for i, len in enumerate(len_vals):
    for j, mat in enumerate(mat_vals):
        R[i, j] = clf.predict_proba(np.array([mat, len]).reshape(1,-1))[:,1]

fig, ax = plt.subplots(figsize=(10, 5.7))

cs1 = ax.contourf(nem_vals, len_vals, R, alpha=0.75, cmap=plt.cm.Greys)
ctr1 = ax.contour(nem_vals, len_vals, R)
plt.clabel(ctr1, inline=1, fontsize=13)
plt.colorbar(cs1, ax=ax)
ax.set_title("Probability of being bad teacher")
ax.set_xlabel("Math Score", fontsize=16)
ax.set_ylabel("Verbal Score", fontsize=16)
plt.savefig(r'C:\\Users\\Franco\\GitHub\\teacher-predictions\\output\\Contour.pdf', bbox_inches='tight')









#Three dimensional scatter:
#==========================

df_rf_test.head(1)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

df_rf_test['random'] = np.random.randn(df_rf_test.shape[0])

ax.scatter(df_rf_test.paaverbal[df_rf_test.random<-2.6], df_rf_test.y_test_hat[df_rf_test.random<-2.6], df_rf_test.nem[df_rf_test.random<-2.6] , c=c, marker=m)
ax.set_xlabel('Verbal Score')
ax.set_ylabel('Bad Teacher Prob')
ax.set_zlabel('GPA Score')

plt.show()





#Then: we should work with RandomForestClassifier

##########################################
#Start with the Score Sensibility analysis
##########################################
from sklearn import tree, svm
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
#CrossValidation Excercise Accuracy tests:
accuracy_type2_rf = []
accuracy_overall_rf = []

accuracy_type2_dt = []
accuracy_overall_dt = []

accuracy_type2_svc = []
accuracy_overall_svc = []

accuracy_type2_lr = []
accuracy_overall_lr = []

# Cross Validate

for i in range(150):

    X_train_transformed, X_test_transformed, y_train, y_test = train_test_split(X_transformed.drop(columns = ['Average', 'paa_avg']), Y, test_size=0.15)

    clf = RandomForestClassifier()
    clf = clf.fit(X_train_transformed, y_train)
    y_test_hat = clf.predict(X_test_transformed)  #Prob of bad Teacher
    acc_overall = np.mean(y_test == y_test_hat)
    acc_type2 = np.mean(y_test[y_test_hat==1] == y_test_hat[y_test_hat==1])
    accuracy_overall_rf.append(acc_overall)
    accuracy_type2_rf.append(acc_type2)

    clf = DecisionTreeClassifier()
    clf = clf.fit(X_train_transformed, y_train)
    y_test_hat = clf.predict(X_test_transformed)  #Prob of bad Teacher
    acc_overall = np.mean(y_test == y_test_hat)
    acc_type2 = np.mean(y_test[y_test_hat==1] == y_test_hat[y_test_hat==1])
    accuracy_overall_dt.append(acc_overall)
    accuracy_type2_dt.append(acc_type2)

    clf = LogisticRegression()
    clf = clf.fit(X_train_transformed, y_train)
    y_test_hat = clf.predict(X_test_transformed)  #Prob of bad Teacher
    acc_overall = np.mean(y_test == y_test_hat)
    acc_type2 = np.mean(y_test[y_test_hat==1] == y_test_hat[y_test_hat==1])
    accuracy_overall_lr.append(acc_overall)
    accuracy_type2_lr.append(acc_type2)

    print(f"Crosvalidation: {i}")

sns.kdeplot(accuracy_overall_rf, alpha=0.8, color = 'orange', label = 'Overall Accuracy')
sns.kdeplot(accuracy_type2_rf, alpha=0.8, color = 'gray', label = 'Type 2 Error')
plt.xlabel('Prediction success')
plt.ylabel('Density')
plt.legend(loc=1)
plt.savefig(r'D:\\Google Drive\\Teacher Quality\\4. Work\\Prediction\\Appendix\\cross_validation_accuracy.pdf', bbox_inches='tight')


sns.kdeplot(accuracy_overall_rf, alpha=0.8, color = 'orange', label = 'Random Forest')
sns.kdeplot(accuracy_overall_dt, alpha=0.8, color = 'gray', label = 'Decision Tree')
sns.kdeplot(accuracy_overall_lr, alpha=0.8, color = 'black', label = 'Logistic Regression')
plt.xlabel('Prediction success')
plt.ylabel('Density')
plt.legend(loc=1)
plt.savefig(r'D:\\Google Drive\\Teacher Quality\\4. Work\\Prediction\\Appendix\\cv_model_selection_std2.pdf', bbox_inches='tight')
plt.savefig(r'D:\\Google Drive\\Teacher Quality\\4. Work\\Prediction\\Appendix\\cv_model_selection.pdf', bbox_inches='tight')

#Plot Type 2 error:
sns.kdeplot(accuracy_type2_rf, alpha=0.8, color = 'orange', label = 'Random Forest')
sns.kdeplot(accuracy_type2_dt, alpha=0.8, color = 'gray', label = 'Decision Tree')
#sns.kdeplot(accuracy_type2_lr, alpha=0.8, color = 'black', label = 'Logistic Regression')
plt.xlabel('Prediction success')
plt.ylabel('Density')
plt.legend(loc=1)
plt.savefig(r'D:\\Google Drive\\Teacher Quality\\4. Work\\Prediction\\Appendix\\cv_model_selection_e2_std2.pdf', bbox_inches='tight')

clf = RandomForestClassifier()
clf = clf.fit(X_train_transformed, y_train)
y_test_hat = clf.predict(X_test_transformed)  #Prob of bad Teacher
acc_overall = np.mean(y_test == y_test_hat)
acc_type2 = np.mean(y_test[y_test_hat==1] == y_test_hat[y_test_hat==1])
accuracy_overall_rf.append(acc_overall)
accuracy_type2_rf.append(acc_type2)

importance = regr_1.feature_importances_
FeatImp = {}

for x in range(0,len(feature_list)):
    if importance[x]>0:
        FeatImp[feature_list[x]]=importance[x]

for key, value in sorted(FeatImp.iteritems(), key=lambda (k,v): (v,k), reverse=True):
    print "%s: %s" % (key, value)
