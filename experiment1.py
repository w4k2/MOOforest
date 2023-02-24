import numpy as np
import os
import time
from joblib import Parallel, delayed
import logging
import traceback
from pathlib import Path
import warnings
from imblearn.metrics import specificity_score

from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone

from methods.MOOforest import MOOforest
from methods.Random_FS import RandomFS
from utils.load_datasets import load_dataset
import math


# Calculate geometric_mean_score based on Precision and Recall
def geometric_mean_score_pr(recall, precision):
    return math.sqrt(recall*precision)

"""
Datasets are from KEEL repository.
"""

base_estimator = DecisionTreeClassifier(random_state=1234)
methods = {
    "MOOforest_promethee":
        MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="promethee" ,criteria_weights=np.array([0.5, 0.5])),
    "MOOforest_recall":
        MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="recall"),
    "MOOforest_precision":
        MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="precision"),
    "DT":
        DecisionTreeClassifier(random_state=1234),
    "RF":
        RandomForestClassifier(n_estimators=15, bootstrap=False, random_state=1234),
    "RF_b":
        RandomForestClassifier(n_estimators=15, bootstrap=True, random_state=1234),
}

# Repeated Stratified K-Fold cross validator
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=111)
n_folds = n_splits * n_repeats

DATASETS_DIR = "datasets/"
# DATASETS_DIR = "datasets_test/"
dataset_paths = []
for root, _, files in os.walk(DATASETS_DIR):
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))

metrics = [
    balanced_accuracy_score,
    f1_score,
    specificity_score,
    recall_score,
    precision_score,
    geometric_mean_score_pr # Gmean based on Precision and Recall
    ]
metrics_alias = [
    "BAC",
    "F1score",
    "Specificity",
    "Recall",
    "Precision",
    "Gmean"]

n_rows_p = 1000

if not os.path.exists("textinfo"):
    os.makedirs("textinfo")
logging.basicConfig(filename='textinfo/experiment1.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
logging.info("--------------------------------------------------------------------------------")
logging.info("-------                        NEW EXPERIMENT                            -------")
logging.info("--------------------------------------------------------------------------------")


def compute(dataset_id, dataset_path):
    logging.basicConfig(filename='textinfo/experiment1.log', filemode="a", format='%(asctime)s - %(levelname)s: %(message)s', level='DEBUG')
    try:
        warnings.filterwarnings("ignore")

        X, y = load_dataset(dataset_path)
        # Normalization - transform data to [0, 1]
        X = MinMaxScaler().fit_transform(X, y)
        scores = np.zeros((len(metrics), len(methods), n_folds))
        pareto_solutions = np.zeros((len(methods), n_folds, n_rows_p, 2))
        # diversity = np.zeros((len(methods), n_folds, 4))
        time_for_all = np.zeros((len(methods), n_folds))
        dataset_name = Path(dataset_path).stem

        print("START: %s" % (dataset_path))
        logging.info("START - %s" % (dataset_path))
        start = time.time()

        for fold_id, (train, test) in enumerate(rskf.split(X, y)):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            for clf_id, clf_name in enumerate(methods):
                start_method = time.time()
                clf = clone(methods[clf_name])
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                # Scores for each metric
                for metric_id, metric in enumerate(metrics):
                    if metric_id >= 1 and metric_id <=4: # if F1score, Specificity, Recall, Precision
                        # average="weighted" : this alters ‘macro’ to account for label imbalance
                        scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred, average="weighted")
                    elif metric_id==5: # if Gmean
                        scores[metric_id, clf_id, fold_id] = metric(scores[3, clf_id, fold_id], scores[4, clf_id, fold_id])
                    else:
                        scores[metric_id, clf_id, fold_id] = metric(y_test, y_pred)

                # # Diversity
                # calculate_diversity = getattr(clf, "calculate_diversity", None)
                # if callable(calculate_diversity):
                #     diversity[clf_id, fold_id] = clf.calculate_diversity()
                # else:
                #     diversity[clf_id, fold_id] = None

                end_method = time.time() - start_method
                logging.info("DONE METHOD %s - %s fold: %d (Time: %f [s])" % (clf_name, dataset_path, fold_id, end_method))
                print("DONE METHOD %s - %s fold: %d (Time: %.2f [s])" % (clf_name, dataset_path, fold_id, end_method))

                time_for_all[clf_id, fold_id] = end_method

                if hasattr(clf, 'solutions'):
                    # print(clf.solutions)
                    for sol_id, solution in enumerate(clf.solutions):
                        for s_id, s in enumerate(solution):
                            pareto_solutions[clf_id, fold_id, sol_id, s_id] = s
                
        # Save results to csv
        for clf_id, clf_name in enumerate(methods):
            # Save metric results
            for metric_id, metric in enumerate(metrics_alias):
                filename = "results/experiment1/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                if not os.path.exists("results/experiment1/raw_results/%s/%s/" % (metric, dataset_name)):
                    os.makedirs("results/experiment1/raw_results/%s/%s/" % (metric, dataset_name))
                np.savetxt(fname=filename, fmt="%f", X=scores[metric_id, clf_id, :])
            # # Save diversity results
            # filename = "results/experiment1/diversity_results/%s/%s_diversity.csv" % (dataset_name, clf_name)
            # if not os.path.exists("results/experiment1/diversity_results/%s/" % (dataset_name)):
            #     os.makedirs("results/experiment1/diversity_results/%s/" % (dataset_name))
            # np.savetxt(fname=filename, fmt="%f", X=diversity[clf_id, :, :])
            # Save time
            filename = "results/experiment1/time_results/%s/%s_time.csv" % (dataset_name, clf_name)
            if not os.path.exists("results/experiment1/time_results/%s/" % (dataset_name)):
                os.makedirs("results/experiment1/time_results/%s/" % (dataset_name))
            np.savetxt(fname=filename, fmt="%f", X=time_for_all[clf_id, :])

        # Save results pareto_solutions to csv
        for clf_id, clf_name in enumerate(methods):
            for fold_id in range(n_folds):
                for sol_id in range(n_rows_p):
                    if (pareto_solutions[clf_id, fold_id, sol_id, 0] != 0.0) and (pareto_solutions[clf_id, fold_id, sol_id, 1] != 0.0):
                        filename_pareto = "results/experiment1/pareto_raw/%s/%s/fold%d/sol%d.csv" % (dataset_name, clf_name, fold_id, sol_id)
                        if not os.path.exists("results/experiment1/pareto_raw/%s/%s/fold%d/" % (dataset_name, clf_name, fold_id)):
                            os.makedirs("results/experiment1/pareto_raw/%s/%s/fold%d/" % (dataset_name, clf_name, fold_id))
                        np.savetxt(fname=filename_pareto, fmt="%f", X=pareto_solutions[clf_id, fold_id, sol_id, :])

        end = time.time() - start
        logging.info("DONE - %s (Time: %d [s])" % (dataset_path, end))
        print("DONE - %s (Time: %d [s])" % (dataset_path, end))

    except Exception as ex:
        logging.exception("Exception in %s" % (dataset_path))
        print("ERROR: %s" % (dataset_path))
        traceback.print_exc()
        print(str(ex))


# Multithread; n_jobs - number of threads, where -1 all threads, safe for my computer 2
Parallel(n_jobs=-1)(
                delayed(compute)
                (dataset_id, dataset_path)
                for dataset_id, dataset_path in enumerate(dataset_paths)
                )
