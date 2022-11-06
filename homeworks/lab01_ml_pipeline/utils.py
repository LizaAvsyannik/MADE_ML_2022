import matplotlib.pyplot as plt
import numpy as np
from scikitplot.metrics import plot_roc
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


RANDOM_STATE = 42


def model_quality(model, X, y):
    y_pred = model.predict(X)
    print(f'Weighted F1 Score {f1_score(y, y_pred, average="weighted")}')
    print(f'Accuracy Score {accuracy_score(y, y_pred)}')

    y_proba = model.predict_proba(X)
    plot_roc(y, y_proba)


def plot_f1_accuracy(f1_scores, accuracy_scores, n_min, n_max, step):
    n_estimators = np.arange(n_min, n_max, step)
    plt.plot(n_estimators, f1_scores, label='F1 Score')
    plt.plot(n_estimators, accuracy_scores, label='Accuracy Score')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Scores')
    plt.ylim((0.0, 1.0))
    plt.xticks(np.arange(min(n_estimators), max(n_estimators) + 1, 5))
    plt.legend()

    f1_scores = np.array(f1_scores)
    best_f1_n = n_estimators[f1_scores.argmax()]
    print(f'Best F1 {f1_scores.max()}, Number of estimators: {best_f1_n}')
    accuracy_scores = np.array(accuracy_scores)
    best_accuracy_n = n_estimators[accuracy_scores.argmax()]
    print(f'Best Accuracy {accuracy_scores.max()}, Number of estimators: {best_accuracy_n}')
    return best_f1_n, best_accuracy_n


def explore_ensemble_size(random_forest_model, X_train, y_train, X_test, y_true, n_min, n_max, step):
    f1_scores = []
    accuracy_scores = []

    for n_estimators in tqdm(range(n_min, n_max, step)):
        model = random_forest_model(n_estimators=n_estimators, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
        accuracy_scores.append(accuracy_score(y_true, y_pred))

    return plot_f1_accuracy(f1_scores, accuracy_scores, n_min, n_max, step)
