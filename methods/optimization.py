import numpy as np
import autograd.numpy as anp
from scipy.stats import mode
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_score, recall_score
from pymoo.core.problem import ElementwiseProblem


class Optimization(ElementwiseProblem):
    def __init__(self, X, y, estimator, n_features, n_classifiers=10, **kwargs):
        self.X = X
        self.y = y
        self.estimator = estimator
        self.n_features = n_features
        self.n_classifiers = n_classifiers
        self.classes_, _ = np.unique(self.y, return_inverse=True)

        # Lower and upper bounds for x - 1d array with length equal to number of variable
        n_variable = self.n_classifiers * self.n_features
        xl_binary = [0] * n_variable
        xu_binary = [1] * n_variable

        super().__init__(n_var=n_variable, n_obj=2,
                         n_constr=0, xl=xl_binary, xu=xu_binary, **kwargs)
        
    def predict(self, X, selected_features, ensemble):
        predictions = np.array([member_clf.predict(X[:, sf]) for member_clf, sf in zip(ensemble, selected_features)])
        prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    def validation(self, x, classes=None):
        
        selected_features = []
        for result_opt in x:
            if result_opt > 0.5:
                feature = True
                selected_features.append(feature)
            else:
                feature = False
                selected_features.append(feature)
        # Wielkość wektora to f x n_clf, np. 7 * 15 = 105
        # print(len(selected_features))

        cross_validation = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=111)
        # Podział pełnego wektora cech na tablicę zawierającą n_clf arrays, gdzie każda array jest dla oddzielnego modelu
        selected_features = np.array_split(selected_features, self.n_classifiers)
        metrics_folds = []
        for fold_id, (train, test) in enumerate(cross_validation.split(self.X, self.y)):
            # Tu musi być wyzerowany ensemble, żeby liczba modeli w nim zawsze była taka jak podajemy na początku, np. 15
            ensemble = []
            for sf in selected_features:
                # If at least one element in sf is True
                if True in sf:
                    X_train = self.X[train]
                    y_train = self.y[train]
                    X_test = self.X[test]
                    y_test = self.y[test]
                    candidate = clone(self.estimator)
                    candidate.fit(X_train[:, sf], y_train)
                    ensemble.append(candidate)
            for index in range(self.n_classifiers):
                # If at least one element in selected_features is True
                if True in selected_features[index]:
                    pass
                else:
                    metrics = [0, 0]
                    return metrics
            y_pred = self.predict(X_test, selected_features, ensemble)
            metrics = [precision_score(y_test, y_pred, average="weighted"), recall_score(y_test, y_pred, average="weighted")]
            metrics_folds.append(metrics)
        mean_score = np.mean(metrics_folds, axis=0)
        std = np.std(metrics_folds, axis=0)
        return mean_score

    def _evaluate(self, x, out, *args, **kwargs):
        scores = self.validation(x)
        # Function F is always minimize, but the minus sign (-) before F means maximize
        # f1 - precision, f2 - recall
        f1 = -1 * scores[0]
        f2 = -1 * scores[1]
        out["F"] = anp.column_stack(np.array([f1, f2]))