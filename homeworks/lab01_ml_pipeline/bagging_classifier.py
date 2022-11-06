import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
from tqdm import tqdm
from utils import plot_f1_accuracy


class CustomBaggingClassifier:
    def __init__(self, base_estimator, n_max_estimators, X, y):
        self.__base_estimator = base_estimator
        self.__n_max_estimators = n_max_estimators
        self.__X = X
        self.__y = y
        self.__models = []
        self.__resample()

    def __resample(self):
        data = np.hstack((self.__X, self.__y[..., None]))
        self.__bootstraped_data = []
        for i in tqdm(range(self.__n_max_estimators)):
            self.__bootstraped_data.append(resample(data))

    def fit(self):
        for i in tqdm(range(self.__n_max_estimators)):
            model = clone(self.__base_estimator)
            X_train = self.__bootstraped_data[i][..., :-1]
            y_train = self.__bootstraped_data[i][..., -1]
            model.fit(X_train, y_train)
            self.__models.append(model)

    def predict(self, X, n_estimators):
        predictions = []
        for i in range(n_estimators):
            predictions.append(self.__models[i].predict_proba(X))
        predictions = np.array(predictions)
        predictions = np.mean(predictions, axis=0)
        predictions = np.argmax(predictions, axis=-1)
        predictions = [self.__models[0].classes_[i] for i in predictions]
        return predictions

    def explore_ensemble_size(self, X, y_true, n_min, n_max, step):
        f1_scores = []
        accuracy_scores = []

        for n_estimators in tqdm(range(n_min, n_max, step)):
            y_pred = self.predict(X, n_estimators)
            f1_scores.append(f1_score(y_true, y_pred, average='weighted'))
            accuracy_scores.append(accuracy_score(y_true, y_pred))

        return plot_f1_accuracy(f1_scores, accuracy_scores, n_min, n_max, step)
