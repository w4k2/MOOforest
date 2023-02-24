import numpy as np
from scipy.stats import mode

from sklearn.base import BaseEstimator, clone
from pymoo.optimize import minimize
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.factory import get_reference_directions

from .optimization import Optimization
from utils.utils_diversity import calc_diversity_measures


class MOOforest(BaseEstimator):
    def __init__(self, base_classifier, n_classifiers=10, predict_decision="MV", n_gen=100, pareto_decision="promethee", criteria_weights=None):
        self.base_classifier = base_classifier
        self.n_classifiers = n_classifiers
        self.selected_features = []
        self.n_gen = n_gen
        self.predict_decision = predict_decision
        self.pareto_decision = pareto_decision
        self.criteria_weights = criteria_weights

    def partial_fit(self, X, y, classes=None):
        self.X, self.y = X, y
        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(self.y, return_inverse=True)
        n_features = X.shape[1]

        # Create optimization problem
        problem = Optimization(X, y, estimator=self.base_classifier, n_classifiers=self.n_classifiers, n_features=n_features)
        
        # 2 - oznacza liczbę M, czyli liczba kryteriów
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=50)
        algorithm = MOEAD(
            ref_dirs=ref_dirs,
            n_neighbors=15,
            prob_neighbor_mating=0.7,
            )

        res = minimize(problem,
                       algorithm,
                       ('n_gen', self.n_gen), # potem daj większą liczbę, np. 200
                       seed=1,
                       save_history=True,
                       verbose=False)
        # Wyjście jeśli verbose=True: 
        # n_gen - The current number of generations or iterations until this point.
        # n_eval - The number of function evaluations so far.
        # n_nds - For multi-objective problems, the number of non-dominated solutions of the optima found.
        # eps/indicator - For multi-objective problems, the change of the indicator (ideal, nadir, f) over the last few generations (only printed if the Pareto-front is unknown). For more information we encourage you to have a look at the corresponding publication ([27], pdf).
        self.res_history = res.history

        # F returns all Pareto front solutions (quality) in form [-precision, -recall]
        self.solutions = res.F        

        # Wybór jednego rozwiązania z Pareto Frontu:
        features_for_ensemble = []
        if self.pareto_decision == 'recall':
            index = np.argmin(self.solutions[:, 1], axis=0)
            features_for_ensemble = res.X[index]
        elif self.pareto_decision == 'precision':
            index = np.argmin(self.solutions[:, 0], axis=0)
            features_for_ensemble = res.X[index]
        elif self.pareto_decision == 'promethee':
            # if only one solution has been found
            if self.solutions.shape[0] == 1:
                index = 0
                features_for_ensemble = res.X[index]
            else:
                # criteria min (0) or max (1) optimization array
                self.criteria_min_max = ([0, 0])
                # u - usual
                self.preference_function = (['u', 'u'])
                net_flows = promethee_function(self.solutions, self.criteria_min_max, self.preference_function, self.criteria_weights)
                # Ranking of the net flows
                index = np.argmax(net_flows, axis=0)
                features_for_ensemble = res.X[index]

        sf_ensemble_binary = []
        for value in features_for_ensemble:
            if value > 0.5:
                feature = True
                sf_ensemble_binary.append(feature)
            else:
                feature = False
                sf_ensemble_binary.append(feature)
        # Podział wektora cech o rozmiarze f x n, na tablicę 2-wymiarową zawierająca osobne wektory dla każdego ensembla
        self.selected_features = np.array_split(sf_ensemble_binary, self.n_classifiers)

        for sf_model in self.selected_features:
            candidate = clone(self.base_classifier).fit(X[:, sf_model], y)
            # Add candidate to the ensemble
            self.ensemble.append(candidate)

        # Pruning based on balanced_accuracy_score
        # if self.pruning:
        #     bac_array = []
        #     for sf, clf in zip(self.selected_features, self.ensemble):
        #         y_pred = clf.predict(X[:, sf])
        #         bac = balanced_accuracy_score(y, y_pred)
        #         bac_array.append(bac)
        #     bac_arg_sorted = np.argsort(bac_array)
        #     self.ensemble_arr = np.array(self.ensemble)
        #     # The percent of deleted models, ex. 0.3 from 10 models = 30 % models will be deleted
        #     pruned = 0.3
        #     pruned_indx = int(pruned * len(self.ensemble))
        #     selected_models = bac_arg_sorted[pruned_indx:]
        #     self.ensemble_arr = self.ensemble_arr[selected_models]
        #     self.ensemble = self.ensemble_arr.tolist()

        #     selected_features_list = [sf.tolist() for sf in self.selected_features]
        #     selected_features_arr = np.array(selected_features_list)
        #     self.selected_features = selected_features_arr[selected_models, :]

        return self

    def fit(self, X, y, classes=None):
        self.ensemble = []
        self.partial_fit(X, y, classes)

    def ensemble_support_matrix(self, X):
        # Ensemble support matrix
        return np.array([member_clf.predict_proba(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])

    def predict(self, X):
        # Prediction based on the Average Support Vectors
        if self.predict_decision == "ASV":
            ens_sup_matrix = self.ensemble_support_matrix(X)
            average_support = np.mean(ens_sup_matrix, axis=0)
            prediction = np.argmax(average_support, axis=1)
        # Prediction based on the Majority Voting
        elif self.predict_decision == "MV":
            predictions = np.array([member_clf.predict(X[:, sf]) for member_clf, sf in zip(self.ensemble, self.selected_features)])
            prediction = np.squeeze(mode(predictions, axis=0)[0])
        return self.classes_[prediction]

    def predict_proba(self, X):
        probas_ = [clf.predict_proba(X) for clf in self.ensemble]
        return np.average(probas_, axis=0)

    # def calculate_diversity(self):
    #     '''
    #     entropy_measure_e: E varies between 0 and 1, where 0 indicates no difference and 1 indicates the highest possible diversity.
    #     kw - Kohavi-Wolpert variance
    #     Q-statistic: <-1, 1>
    #     Q = 0 statistically independent classifiers
    #     Q < 0 classifiers commit errors on different objects
    #     Q > 0 classifiers recognize the same objects correctly
    #     '''
    #     if len(self.ensemble) > 1:
    #         # All measures for whole ensemble
    #         self.entropy_measure_e, self.k0, self.kw, self.disagreement_measure, self.q_statistic_mean = calc_diversity_measures(self.X, self.y, self.ensemble, self.selected_features, p=0.01)

    #         return(self.entropy_measure_e, self.kw, self.disagreement_measure, self.q_statistic_mean)


    # Calculation uni weighted to promethee method
def uni_cal(solutions_col, criteria_min_max, preference_function, criteria_weights):
    uni = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    uni_weighted = np.zeros((solutions_col.shape[0], solutions_col.shape[0]))
    for i in range(np.size(uni, 0)):
        for j in range(np.size(uni, 1)):
            if i == j:
                uni[i, j] = 0
            # Usual preference function
            elif preference_function == 'u':
                diff = solutions_col[j] - solutions_col[i]
                if diff > 0:
                    uni[i, j] = 1
                else:
                    uni[i, j] = 0
            uni_weighted[i][j] = criteria_weights * uni[i, j]
    # criteria min (0) or max (1) optimization array
    if criteria_min_max == 0:
        uni_weighted = uni_weighted
    elif criteria_min_max == 1:
        uni_weighted = uni_weighted.T
    return uni_weighted


# promethee method to choose one solution from the pareto front
def promethee_function(solutions, criteria_min_max, preference_function, criteria_weights):
    weighted_unis = []
    for i in range(solutions.shape[1]):
        weighted_uni = uni_cal(solutions[:, i:i + 1], criteria_min_max[i], preference_function[i], criteria_weights[i])
        weighted_unis.append(weighted_uni)
    agregated_preference = []
    uni_acc = weighted_unis[0]
    uni_cost = weighted_unis[1]
    # Combine two criteria into agregated_preference
    for (item1, item2) in zip(uni_acc, uni_cost):
        agregated_preference.append((item1 + item2)/sum(criteria_weights))
    agregated_preference = np.array(agregated_preference)

    n_solutions = agregated_preference.shape[0] - 1
    # Sum by rows - positive flow
    pos_flows = []
    pos_sum = np.sum(agregated_preference, axis=1)
    for element in pos_sum:
        pos_flows.append(element/n_solutions)
    # Sum by columns - negative flow
    neg_flows = []
    neg_sum = np.sum(agregated_preference, axis=0)
    for element in neg_sum:
        neg_flows.append(element/n_solutions)
    # Calculate net_flows
    net_flows = []
    for i in range(len(pos_flows)):
        net_flows.append(pos_flows[i] - neg_flows[i])
    return net_flows