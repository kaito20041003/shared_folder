import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def load_and_preprocess_data():
    iris = datasets.load_iris()
    print(iris.DESCR)
    print("Target Names:", iris.target_names)

    # 特徴量と目的変数の選択（setosa を除く）
    X = iris.data[:, 2:4]
    y = (iris.target == 2).astype(int)
    mask = iris.target != 0
    X, y = X[mask], y[mask]

    # 訓練・テスト分割と標準化
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, C=1, seed=42):
    # SVC
    svc = SVC(kernel='linear', C=C, random_state=seed)
    svc.fit(X_train, y_train)
    svc_pred = svc.predict(X_test)
    svc_acc = accuracy_score(y_test, svc_pred)
    print(f'SVC accuracy score (C={C}): {svc_acc:.2f}')

    # Logistic Regression
    log_reg = LogisticRegression(C=C, random_state=seed)
    log_reg.fit(X_train, y_train)
    log_pred = log_reg.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)
    print(f'Logistic Regression accuracy score (C={C}): {log_acc:.2f}')

    return svc, log_reg

def plot_decision_boundaries(svc_model, log_model, X_test, y_test, C):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    models = [(svc_model, ax[0], 'SVC'), (log_model, ax[1], 'Logistic Regression')]

    for model, axis, title in models:
        axis.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap=plt.cm.Paired)
        xlim = axis.get_xlim()
        ylim = axis.get_ylim()
        xx, yy = np.linspace(*xlim, 30), np.linspace(*ylim, 30)
        YY, XX = np.meshgrid(yy, xx)
        grid = np.vstack([XX.ravel(), YY.ravel()]).T

        if isinstance(model, SVC):
            Z = model.decision_function(grid).reshape(XX.shape)
            axis.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'], alpha=0.5)
        else:
            Z = model.predict_proba(grid)[:, 1].reshape(XX.shape)
            axis.contour(XX, YY, Z, levels=[0.5], colors='k', linewidths=2)

        axis.set_title(f'{title} (C={C})')

    plt.show()

def main():
    C = 1
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    svc_model, log_model = train_and_evaluate_models(X_train, X_test, y_train, y_test, C)
    plot_decision_boundaries(svc_model, log_model, X_test, y_test, C)

if __name__ == "__main__":
    main()
