import os
import numpy as np
import matplotlib.pyplot as plt
from .load_datasets import load_dataset
from .datasets_table_description import calc_imbalance_ratio



def result_tables_IR_precision_recall(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    imbalance_ratios = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)
    IR_argsorted = np.argsort(imbalance_ratios)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables_IR_prrec/" % experiment_name):
            os.makedirs("results/%s/tables_IR_prrec/" % experiment_name)
        with open("results/%s/tables_IR_prrec/results_%s.tex" % (experiment_name, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{Precision and Recall}", file=file)
            columns = ""
            for i in methods:
                columns += " c|"
            columns = columns[:-1]
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{V{3}rV{3}%sV{3}%sV{3}}" % (columns, columns), file=file)
            print("\\hlineB{3}", file=file)
            first_row = " & \multicolumn{%d}{cV{3}}{\\textbf{Recall}} & \multicolumn{%d}{cV{3}}{\\textbf{Precision}} \\\\" % (len(methods), len(methods))
            print(first_row, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{ID} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(IR_argsorted):
                id += 1
                line = "\\textbf{%d}" % (id)
                # lineir = "$%s$" % (dataset_paths[arg])
                # print(line, lineir)
                line_values_recall = []
                line_values_recall = mean_scores[arg, 3, :]
                max_value_recall = np.amax(line_values_recall)
                line_values_precision = []
                line_values_precision = mean_scores[arg, 4, :]
                max_value_precision = np.amax(line_values_precision)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, 3, clf_id] == max_value_recall:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, 3, clf_id], stds[arg, 3, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, 3, clf_id], stds[arg, 3, clf_id])

                    if mean_scores[arg, 4, clf_id] == max_value_precision:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, 4, clf_id], stds[arg, 4, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, 4, clf_id], stds[arg, 4, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\hlineB{3}", file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)



def result_tables_IR(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    imbalance_ratios = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        IR = calc_imbalance_ratio(X, y)
        imbalance_ratios.append(IR)
    IR_argsorted = np.argsort(imbalance_ratios)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables_IR/" % experiment_name):
            os.makedirs("results/%s/tables_IR/" % experiment_name)
        with open("results/%s/tables_IR/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{ID} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(IR_argsorted):
                id += 1
                line = "%d" % (id)
                # lineir = "$%s$" % (dataset_paths[arg])
                # print(line, lineir)
                line_values = []
                line_values = mean_scores[arg, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)


def result_tables_features(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    X_features = []
    for dataset_path in dataset_paths:
        X, y = load_dataset(dataset_path)
        X_features.append(X.shape[1])
    X_features_argsorted = np.argsort(X_features)
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables_features/" % experiment_name):
            os.makedirs("results/%s/tables_features/" % experiment_name)
        with open("results/%s/tables_features/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"
            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{ID} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for id, arg in enumerate(X_features_argsorted):
                id += 1
                line = "%d" % (id)
                # lineir = "$%s$" % (dataset_paths[arg])
                # print(line, lineir)
                line_values = []
                line_values = mean_scores[arg, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[arg, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[arg, metric_id, clf_id], stds[arg, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)


def result_tables(dataset_paths, metrics_alias, mean_scores, methods, stds, experiment_name):
    for metric_id, metric in enumerate(metrics_alias):
        if not os.path.exists("results/%s/tables/" % experiment_name):
            os.makedirs("results/%s/tables/" % experiment_name)
        with open("results/%s/tables/results_%s_%s.tex" % (experiment_name, metric, experiment_name), "w+") as file:
            print("\\begin{table}[!ht]", file=file)
            print("\\centering", file=file)
            print("\\caption{%s}" % (metric), file=file)
            columns = "r"
            for i in methods:
                columns += " c"

            print("\\scalebox{0.4}{", file=file)
            print("\\begin{tabular}{%s}" % columns, file=file)
            print("\\hline", file=file)
            columns_names = "\\textbf{Dataset name} &"
            for name in methods:
                name = name.replace("_", "-")
                columns_names += f'\\textbf{{{name}}} & '
            columns_names = columns_names[:-3]
            columns_names += "\\\\"
            print(columns_names, file=file)
            print("\\hline", file=file)
            for dataset_id, dataset_path in enumerate(dataset_paths):
                line = "$%s$" % (dataset_path)
                line_values = []
                line_values = mean_scores[dataset_id, metric_id, :]
                max_value = np.amax(line_values)
                for clf_id, clf_name in enumerate(methods):
                    if mean_scores[dataset_id, metric_id, clf_id] == max_value:
                        line += " & \\textbf{%0.3f $\\pm$ %0.3f}" % (mean_scores[dataset_id, metric_id, clf_id], stds[dataset_id, metric_id, clf_id])
                    else:
                        line += " & %0.3f $\\pm$ %0.3f" % (mean_scores[dataset_id, metric_id, clf_id], stds[dataset_id, metric_id, clf_id])
                line += " \\\\"
                print(line, file=file)
            print("\\end{tabular}}", file=file)
            print("\\end{table}", file=file)


def result_tables_for_time(dataset_names, imbalance_ratios, sum_times, methods, experiment_name):
    IR_argsorted = np.argsort(imbalance_ratios)
    if not os.path.exists("results/%s/tables/" % experiment_name):
        os.makedirs("results/%s/tables/" % experiment_name)
    with open("results/%s/tables/time_%s.tex" % (experiment_name, experiment_name), "w+") as file:
        print("\\begin{table}[!ht]", file=file)
        print("\\centering", file=file)
        print("\\caption{Time [s]}", file=file)
        columns = "r"
        for i in methods:
            columns += " c"

        print("\\scalebox{0.4}{", file=file)
        print("\\begin{tabular}{%s}" % columns, file=file)
        print("\\hline", file=file)
        columns_names = "\\textbf{ID} &"
        for name in methods:
            name = name.replace("_", "-")
            columns_names += f'\\textbf{{{name}}} & '
        columns_names = columns_names[:-3]
        columns_names += "\\\\"
        print(columns_names, file=file)
        print("\\hline", file=file)
    
        for id, arg in enumerate(IR_argsorted):
            id += 1
            # ds_name = dataset_names[arg].replace("_", "\\_")
            # line = "%d & \\emph{%s}" % (id, ds_name)
            line = "%d" % (id)
            for clf_id, clf_name in enumerate(methods):
                line += " & %0.3f" % (sum_times[arg, clf_id])
            line += " \\\\"
            print(line, file=file)
        print("\\end{tabular}}", file=file)
        print("\\end{table}", file=file)


# Plot scatter of pareto front solutions and all methods
def scatter_plot(datasets, n_folds, experiment_name, methods, raw_data):
    n_rows_p = 1000
    for dataset_id, dataset in enumerate(datasets):
        print(dataset)
        for fold_id in range(n_folds):
            solutions_moo = []
            for sol_id in range(n_rows_p):
                try:
                    filename_pareto_semoos = "results/%s/pareto_raw/%s/MOOforest/fold%d/sol%d.csv" % (experiment_name, dataset, fold_id, sol_id)
                    solution_moo = np.genfromtxt(filename_pareto_semoos, dtype=np.float32)
                    solution_moo = solution_moo.tolist()
                    solution_moo[0] = solution_moo[0] * (-1)
                    solution_moo[1] = solution_moo[1] * (-1)
                    solutions_moo.append(solution_moo)
                except IOError:
                    pass
            if solutions_moo:
            #  and solutions_semoosb and solutions_semoosbp:
                filename_pareto_chart = "results/%s/scatter_plots/%s/scatter_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
                if not os.path.exists("results/%s/scatter_plots/%s/" % (experiment_name, dataset)):
                    os.makedirs("results/%s/scatter_plots/%s/" % (experiment_name, dataset))

                moo_x = []
                moo_y = []
                for solution in solutions_moo:
                    moo_x.append(solution[0])
                    moo_y.append(solution[1])
                moo_x = np.array(moo_x)
                moo_y = np.array(moo_y)
                plt.grid(True, color="silver", linestyle=":", axis='both')

                # MOOforest pareto
                plt.scatter(moo_x, moo_y, color='darkgray', marker="o", label="MOOforest PF")
                # Precision
                moo_precision = raw_data[dataset_id, 4, 0, fold_id]
                # Recall
                moo_recall = raw_data[dataset_id, 3, 0, fold_id]
                plt.scatter(moo_precision, moo_recall, color='black', marker="o", label="MOOforest")

                # DT
                plt.scatter(raw_data[dataset_id, 4, 1, fold_id], raw_data[dataset_id, 3, 1, fold_id], color='tab:pink', marker=">", label="DT")
                # DE_Forest
                plt.scatter(raw_data[dataset_id, 4, 2, fold_id], raw_data[dataset_id, 3, 2, fold_id], color='tab:blue', marker="+", label="DE_Forest")
                # RandomFS
                plt.scatter(raw_data[dataset_id, 4, 3, fold_id], raw_data[dataset_id, 3, 3, fold_id], color='tab:red', marker="^", label="RandomFS")
                # RandomFS_b
                plt.scatter(raw_data[dataset_id, 4, 4, fold_id], raw_data[dataset_id, 3, 4, fold_id], color='tab:purple', marker="<", label="RandomFS_b")
                # # RF
                # plt.scatter(raw_data[dataset_id, 4, 5, fold_id], raw_data[dataset_id, 3, 5, fold_id], color='tab:blue', marker="v", label="RF")
                # # RF_b
                # plt.scatter(raw_data[dataset_id, 4, 6, fold_id], raw_data[dataset_id, 3, 6, fold_id], color='tab:orange', marker="v", label="RF_b")

                # plt.title("Objective Space", fontsize=12)
                plt.xlabel('Precision', fontsize=12)
                plt.ylabel('Recall', fontsize=12)
                plt.xlim([0, 1.1])
                plt.ylim([0, 1.1])
                plt.legend(loc="best")
                plt.gcf().set_size_inches(9, 6)
                plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                plt.clf()
                plt.close()


# Plot scatter of pareto front solutions and all methods
def scatter_plot_closer(datasets, n_folds, experiment_name, methods, raw_data):
    n_rows_p = 1000
    for dataset_id, dataset in enumerate(datasets):
        print(dataset)
        for fold_id in range(n_folds):
            solutions_moo = []
            for sol_id in range(n_rows_p):
                try:
                    filename_pareto_semoos = "results/%s/pareto_raw/%s/MOOforest/fold%d/sol%d.csv" % (experiment_name, dataset, fold_id, sol_id)
                    solution_moo = np.genfromtxt(filename_pareto_semoos, dtype=np.float32)
                    solution_moo = solution_moo.tolist()
                    solution_moo[0] = solution_moo[0] * (-1)
                    solution_moo[1] = solution_moo[1] * (-1)
                    solutions_moo.append(solution_moo)
                except IOError:
                    pass
            if solutions_moo:
                filename_pareto_chart = "results/%s/scatter_plots_closer/%s/scatter_%s_fold%d" % (experiment_name, dataset, dataset, fold_id)
                if not os.path.exists("results/%s/scatter_plots_closer/%s/" % (experiment_name, dataset)):
                    os.makedirs("results/%s/scatter_plots_closer/%s/" % (experiment_name, dataset))

                moo_x = []
                moo_y = []
                for solution in solutions_moo:
                    moo_x.append(solution[0])
                    moo_y.append(solution[1])
                moo_x = np.array(moo_x)
                moo_y = np.array(moo_y)

                plt.grid(True, color="silver", linestyle=":", axis='both')

                # MOOforest pareto
                plt.scatter(moo_x, moo_y, color='darkgray', marker="o", label="MOOforest PF")
                # Precision
                moo_precision = raw_data[dataset_id, 4, 0, fold_id]
                # Recall
                moo_recall = raw_data[dataset_id, 3, 0, fold_id]
                plt.scatter(moo_precision, moo_recall, color='black', marker="o", label="MOOforest")

                # DT
                plt.scatter(raw_data[dataset_id, 4, 1, fold_id], raw_data[dataset_id, 3, 1, fold_id], color='tab:pink', marker=">", label="DT")
                # RF
                plt.scatter(raw_data[dataset_id, 4, 2, fold_id], raw_data[dataset_id, 3, 2, fold_id], color='tab:blue', marker="v", label="RF")
                # RF_b
                plt.scatter(raw_data[dataset_id, 4, 3, fold_id], raw_data[dataset_id, 3, 3, fold_id], color='tab:orange', marker="v", label="RF_b")
                # DE_Forest
                plt.scatter(raw_data[dataset_id, 4, 4, fold_id], raw_data[dataset_id, 3, 4, fold_id], color='tab:blue', marker="+", label="DE_Forest")
                # RandomFS
                plt.scatter(raw_data[dataset_id, 4, 5, fold_id], raw_data[dataset_id, 3, 5, fold_id], color='tab:red', marker="^", label="RandomFS")
                # RandomFS_b
                plt.scatter(raw_data[dataset_id, 4, 6, fold_id], raw_data[dataset_id, 3, 6, fold_id], color='tab:purple', marker="<", label="RandomFS_b")

                plt.xlabel('Precision', fontsize=12)
                plt.ylabel('Recall', fontsize=12)
                # plt.xlim([0, 1.1])
                # plt.ylim([0, 1.1])
                plt.legend(loc="best")
                plt.gcf().set_size_inches(9, 6)
                plt.savefig(filename_pareto_chart+".png", bbox_inches='tight')
                plt.savefig(filename_pareto_chart+".eps", format='eps', bbox_inches='tight')
                plt.clf()
                plt.close()