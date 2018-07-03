import pandas as pd
import numpy as np
import os
import seaborn as sns
#matplotlib inline
from matplotlib import pyplot
import matplotlib.pyplot as plt
import plotly.plotly as py
import numpy as np
import pylab as plt
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
df.head(1)






from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

n = 100

dft = df[['paa_verbal', 'paa_matematica', 'pf_pje']].dropna()
df['xtile'] = pd.qcut(df.pf_pje, 10, labels = ['percentil: '+str(i) for i in range(10)])
df['worst'] = (df['xtile'] == 'percentil: 0' ) | (df['xtile'] == 'percentil: 1') | (df['xtile'] == 'percentil: 2')

dft = (dft - dft.mean())/dft.std()
dft['random'] = np.random.randn(dft.shape[0])

ax.scatter(dft.paa_verbal[dft.random<-2.6][df.worst == 1], dft.paa_matematica[dft.random<-2.6][df.worst == 1], dft.pf_pje[dft.random<-2.6][df.worst == 1], color = 'r')
ax.scatter(dft.paa_verbal[dft.random<-2.6][df.worst == 0], dft.paa_matematica[dft.random<-2.6][df.worst == 0], dft.pf_pje[dft.random<-2.6][df.worst == 0], color = 'b')
ax.set_xlabel('Verbal Score')
ax.set_ylabel('Math Score')
ax.set_zlabel('Portfolio Score')

plt.show()
