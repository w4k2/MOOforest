import os
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from methods.MOOforest import MOOforest
# from methods.Random_FS import RandomFS
from utils.wilcoxon_ranking import pairs_metrics_multi_grid_all, pairs_metrics_multi_line
from utils.plots import result_tables, result_tables_for_time, result_tables_IR, result_tables_features, scatter_plot, scatter_plot_closer, result_tables_IR_precision_recall
from utils.datasets_table_description import make_description_table
from utils.load_datasets import load_dataset
from utils.datasets_table_description import calc_imbalance_ratio

base_estimator = DecisionTreeClassifier(random_state=1234)
methods = {
    # "MOOforest_promethee":
    #     MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="promethee" ,criteria_weights=np.array([0.5, 0.5])),
    # "MOOforest_recall":
    #     MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="recall"),
    # "MOOforest_precision":
    #     MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="precision"),
    "MOOforest":
        MOOforest(base_classifier=base_estimator, n_classifiers=15, n_gen=200, pareto_decision="promethee" ,criteria_weights=np.array([0.5, 0.5])),
    "DT":
        DecisionTreeClassifier(random_state=1234),
    # "RF":
    #     RandomForestClassifier(n_estimators=15, bootstrap=False, random_state=1234),
    # "RF_b":
    #     RandomForestClassifier(n_estimators=15, bootstrap=True, random_state=1234),

    "DE_Forest":
        base_estimator,
    "RandomFS":
        base_estimator,
    "RandomFS_b":
        base_estimator,
}
method_names = list(methods.keys())

metrics_alias = [
    "BAC",
    "F1score",
    "Specificity",
    "Recall",
    "Precision",
    "Gmean"]

# DATASETS_DIR = "datasets/"
DATASETS_DIR = "datasets_36/"
dataset_paths = []
dataset_names = []
imbalance_ratios = []
for root, _, files in os.walk(DATASETS_DIR):
    print(root, files)
    for filename in filter(lambda _: _.endswith('.dat'), files):
        dataset_paths.append(os.path.join(root, filename))
        dataset_path = os.path.join(root, filename)
        dataset_name = Path(dataset_path).stem
        dataset_names.append(dataset_name)
        X, y = load_dataset(dataset_path)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)

n_splits = 2
n_repeats = 5
n_folds = n_splits * n_repeats
n_methods = len(methods)
n_metrics = len(metrics_alias)
n_datasets = len(dataset_paths)
# Load data from file
data_np = np.zeros((n_datasets, n_metrics, n_methods, n_folds))
mean_scores = np.zeros((n_datasets, n_metrics, n_methods))
stds = np.zeros((n_datasets, n_metrics, n_methods))
sum_times = np.zeros((n_datasets, len(methods)))

for dataset_id, dataset_path in enumerate(dataset_paths):
    dataset_name = Path(dataset_path).stem
    for clf_id, clf_name in enumerate(methods):
        for metric_id, metric in enumerate(metrics_alias):
            try:
                if clf_name == "MOOforest" or clf_name == "DT" or clf_name == "RF" or clf_name == "RF_b":
                    filename = "results/experiment1/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                    if not os.path.isfile(filename):
                        print("File not exist - %s" % filename)
                        # continue
                    scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    data_np[dataset_id, metric_id, clf_id] = scores
                    mean_score = np.mean(scores)
                    mean_scores[dataset_id, metric_id, clf_id] = mean_score
                    std = np.std(scores)
                    stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset_name, clf_name, metric)
            try:
                if clf_name == "DE_Forest" or clf_name == "RandomFS" or clf_name == "RandomFS_b":
                    filename = "/home/joannagrzyb/work/DE-Forest/results/experiment1/raw_results/%s/%s/%s.csv" % (metric, dataset_name, clf_name)
                    if not os.path.isfile(filename):
                        print("File not exist - %s" % filename)
                        # continue
                    scores = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
                    data_np[dataset_id, metric_id, clf_id] = scores
                    mean_score = np.mean(scores)
                    mean_scores[dataset_id, metric_id, clf_id] = mean_score
                    std = np.std(scores)
                    stds[dataset_id, metric_id, clf_id] = std
            except:
                print("Error loading data!", dataset_name, clf_name, metric)
        # try:
        #     filename = "results/experiment1/time_results/%s/%s_time.csv" % (dataset_name, clf_name)
        #     if not os.path.isfile(filename):
        #         # print("File not exist - %s" % filename)
        #         continue
        #     times = np.genfromtxt(filename, delimiter=',', dtype=np.float32)
        #     sum_times[dataset_id, clf_id] = sum(times)
        # except:
        #     print("Error loading time data!", dataset_name, clf_name)

# All datasets with description in the table
# make_description_table(DATASETS_DIR)

experiment_name = "experiment1"
# Results in form of one .tex table of each metric
# result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of each metric sorted by IR
result_tables_IR(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of metrics Precision and Recall sorted by IR
# result_tables_IR_precision_recall(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Results in form of one .tex table of each metric sorted by number of features
# result_tables_features(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name)

# Wilcoxon ranking line - statistic test for my method vs the remaining methods
# pairs_metrics_multi_line(method_names=list(method_names), data_np=data_np, experiment_name=experiment_name, dataset_paths=dataset_paths, metrics=metrics_alias, filename="ex1_wilcoxon", ref_methods=list(method_names))

# Time results in form of .tex table
# result_tables_for_time(dataset_names, imbalance_ratios, sum_times, methods, experiment_name)

# Plot scatter with all methods
# scatter_plot(datasets=dataset_names, n_folds=n_folds, experiment_name=experiment_name, methods=methods, raw_data=data_np)

# Plot scatter with all methods fo only one dataset
# scatter_plot_closer(datasets=dataset_names, n_folds=n_folds, experiment_name=experiment_name, methods=methods, raw_data=data_np)

# Results from 13_02_23 are the final results
