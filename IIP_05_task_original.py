import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

%matplotlib inline

# irisデータセットのロード
iris = datasets.load_iris()
#アヤメのデータ説明
print(iris.DESCR)
#アヤメの種類の説明(['setosa' 'versicolor' 'virginica'])
print(iris.target_names)

# 特徴量として花びらの長さを選択(2列目を取得)
X = iris.data[:, 2:4]
X = X[(iris.target != 0)]
# バージニカ種かそれ以外かの2クラス分類のための目的変数を作成
y = (iris.target == 2).astype(np.int64)
y = y[(iris.target != 0)]
# データを訓練セットとテストセットに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)
# 特徴量のスケーリング
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#ランダムシード
seed = 42

#C(コスト)パラメータのリスト
C = 1

#SVC学習
svc_model = SVC(kernel='linear', C=C, random_state=seed)
svc_model.fit(X_train, y_train)
svc_y_pred = svc_model.predict(X_test)
print('SVC accuracy score for C = {}:'.format(C), accuracy_score(svc_y_pred, y_test))

#ロジスティクス回帰　学習
log_reg_model = LogisticRegression(C=C, random_state=seed)
log_reg_model.fit(X_train, y_train)
log_reg_y_pred = log_reg_model.predict(X_test)
print('LogisticRegression accuracy score for C = {}:'.format(C),accuracy_score(log_reg_y_pred, y_test))
fig, ax = plt.subplots(1,2, figsize=(14,6))

#SVC プロット
ax[0].scatter(X_test[:,0], X_test[:,1], c=y_test, s=30, cmap=plt.cm.Paired)
xlim = ax[0].get_xlim()
ylim = ax[0].get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svc_model.decision_function(xy).reshape(XX.shape)
ax[0].contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
ax[0].set_title('SVC with C = {}'.format(C))

#Logistitc Regression プロット
ax[1].scatter(X_test[:,0], X_test[:,1], c=y_test, s=30, cmap=plt.cm.Paired)
xlim = ax[1].get_xlim()
ylim = ax[1].get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy,xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = log_reg_model.predict_proba(xy)[:,1]
ax[1].contour(XX, YY, Z.reshape(XX.shape), [0.5], linewidths=2, colors='k')
ax[1].set_title('Logistic Regression with C = {}'.format(C))
plt.show()
