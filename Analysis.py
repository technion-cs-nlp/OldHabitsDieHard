import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, stats
import subprocess
import matplotlib.pyplot as plt
import random
from typing import List, Tuple
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.lines as mlines
import matplotlib as mpl

import json
import os
from sklearn.metrics import roc_curve, auc

mpl.rc('font', family='serif', serif=['STIXGeneral'])


def procrustes_2d(X, Y):
    """
    Find optimal 2D rotation matrix R such that Y ≈ R @ X

    Args:
        X: 2 x N matrix (source vectors as columns)
        Y: 2 x N matrix (target vectors as columns)

    Returns:
        theta: rotation angle in radians
    """
    # Compute cross-covariance matrix
    C = np.array(Y).T @ np.array(X)
    # Extract elements
    a, b, c, d = C[0, 0], C[0, 1], C[1, 0], C[1, 1]

    # Compute optimal rotation angle
    theta = np.arctan2(c - b, a + d)

    return theta


def theta_per_example(X, Y):
    """
    Compute rotation angle theta for each pair of 2D vectors in X and Y

    Args:
        X: list of 2D vectors (source)
        Y: list of 2D vectors (target)

    Returns:
        thetas: list of rotation angles in radians
    """
    thetas = []
    for x, y in zip(X, Y):
        cos_theta = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        thetas.append(np.arccos(np.clip(cos_theta, -1, 1)))
    assert len(thetas) == len(X), f"Mismatch in number of angles computed: {len(thetas)} vs {len(X)}"
    return thetas


def CCW_needed_check(v1, v2, eps=1e-12):
    """Determines the signed angle direction between two vectors"""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if np.linalg.norm(v1) < eps or np.linalg.norm(v2) < eps:
        raise ValueError("zero-length vector")

    e1 = v1 / np.linalg.norm(v1)

    # pick a reference r not collinear with e1
    n = e1.size
    # try standard basis vectors until one works
    r = np.ones(n) / np.sqrt(n)
    # build e2 as r projected orthogonal to e1
    temp = r - np.dot(e1, r) * e1
    norm_temp = np.linalg.norm(temp)
    if norm_temp < eps:
        raise RuntimeError("failed to find reference orthogonal component")
    e2 = temp / norm_temp
    # coordinates of v2 in basis (e1, e2)
    x = np.dot(e1, v2)
    y = np.dot(e2, v2)
    theta = np.arctan2(y, x)  # signed angle in (-pi, pi]
    return theta


def plot_lambda(degrees_per_file, split_models=None, ordered="False"):
    """Plots the exponential decay of the error term (Lambda)."""
    for s in split_models:
        lambdas = {}
        for file, obs in degrees_per_file.items():
            if not any([s_i in file for s_i in s]) or ordered not in file:
                continue
            lambda2 = (obs['2-2']["F-F"] + obs['2-2']["H-H"]) - 1
            lambdas[file.split("_")[2] + "_" + file.split("_")[3]] = lambda2
        # plot the lambdas
        plot_lambda_custom_convergence(lambdas, f'plots/lambda_convergence_{s[0]}_ordered_{ordered}.pdf')


def plot_lambda_custom_convergence(lambdas: dict, path):
    # Setup
    time_steps = np.arange(0, 20, 1)  # Look at first 500 steps

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.family': 'serif', 'font.size': 12})

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#F0E442', '#E69F00']

    # Distinct markers for accessibility
    markers = ['o', 's', '^', 'D', 'v', 'P', '*']

    for i, (name, lam) in enumerate(lambdas.items()):
        # Calculate the error decay: Error ~ lambda^t
        # We start from an initial error of 1.0 (normalized)
        decay_curve = lam ** time_steps

        plt.plot(time_steps, decay_curve,
                 label=name.replace("do_not", "Do-Not-Answer").replace("natural_100", "NaturalQA").replace(
                     "triviaqa_100", "TriviaQA").replace("sorry_100", "Sorry").replace("sycophancy_negative",
                                                                                       "S-neg").replace("sycophancy",
                                                                                                        "S-pos").replace(
                     "_100", ""),
                 color=colors[i], marker=markers[i],
                 linewidth=2.5)

    plt.yscale('log')  # Crucial: Makes exponential decay look linear
    plt.xscale('log')
    plt.ylim(10 ** -10, 1)
    plt.xlim(1, 10)
    plt.xlabel('Time (Iterations)', fontsize=30)
    if "llama" in path:
        plt.ylabel('Distance to Stationarity', fontsize=27)
        plt.legend(fontsize=25, frameon=True)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()

    plt.savefig(path, format='pdf', dpi=300)
    plt.close()


def plot_hidden_states_through_time(hidden_states_list: List[List[Tuple]], calculate_angle_per_point: bool = False):
    """
    Calculate the geometric values.
    """
    degrees = {}
    hidden_states_to_test_on = random.sample(range(len(hidden_states_list)), int(len(hidden_states_list) * 0.5))
    hidden_states_to_train_on = [i for i in range(len(hidden_states_list)) if i not in hidden_states_to_test_on]
    hall_mean_vector = np.mean([np.array(hidden_states_list[i][j][0]) for i in hidden_states_to_train_on for j in
                                range(len(hidden_states_list[i])) if hidden_states_list[i][j][1] == 1], axis=0)
    factual_mean_vector = np.mean([np.array(hidden_states_list[i][j][0]) for i in hidden_states_to_train_on for j in
                                   range(len(hidden_states_list[i])) if hidden_states_list[i][j][1] == 0], axis=0)
    hall_mean_vector = hall_mean_vector / np.linalg.norm(hall_mean_vector)
    factual_mean_vector = factual_mean_vector / np.linalg.norm(factual_mean_vector)
    all_test_examples = [hidden_states_list[i] for i in hidden_states_to_test_on]

    def plot_vectors_2d(points_to_plot, hall_mean_vector, factual_mean_vector):
        import numpy as np
        # Normalize the mean vectors
        f_vec = factual_mean_vector / np.linalg.norm(factual_mean_vector)
        h_vec = hall_mean_vector / np.linalg.norm(hall_mean_vector)
        basis_1 = f_vec
        # Second basis vector: orthogonal to first, in the plane of both mean vectors
        basis_2 = h_vec - np.dot(h_vec, basis_1) * basis_1
        basis_2 = basis_2 / np.linalg.norm(basis_2)
        theta_with_sign = np.sign(CCW_needed_check(f_vec, h_vec))
        # Enforce convention: phenomena mean should have POSITIVE B2 component
        if theta_with_sign < 0:
            basis_2 = -basis_2

        # Project all state vectors onto the 2D plane
        points_2d = []
        colors = []
        labels = []

        for i, (state_vector, label) in enumerate(points_to_plot):
            # Normalize and project the state vector
            state_vec_normalized = np.array(state_vector)
            x = np.dot(state_vec_normalized, basis_1)
            y = np.dot(state_vec_normalized, basis_2)

            points_2d.append((x, y))
            colors.append('red' if label == 1 else 'blue')
            labels.append(i)
        return points_2d

    H_F_X = []
    H_F_Y = []
    F_F_X = []
    F_F_Y = []
    H_H_X = []
    H_H_Y = []
    F_H_X = []
    F_H_Y = []
    for test in all_test_examples:
        points_2d = plot_vectors_2d(test, hall_mean_vector, factual_mean_vector)
        for i in range(len(points_2d) - 1):
            if test[i][1] == 1 and test[i + 1][1] == 0:
                H_F_X.append(points_2d[i])
                H_F_Y.append(points_2d[i + 1])
            elif test[i][1] == 0 and test[i + 1][1] == 0:
                F_F_X.append(points_2d[i])
                F_F_Y.append(points_2d[i + 1])
            elif test[i][1] == 1 and test[i + 1][1] == 1:
                H_H_X.append(points_2d[i])
                H_H_Y.append(points_2d[i + 1])
            elif test[i][1] == 0 and test[i + 1][1] == 1:
                F_H_X.append(points_2d[i])
                F_H_Y.append(points_2d[i + 1])
    if calculate_angle_per_point:
        degrees['H-F per point'] = np.degrees(theta_per_example(H_F_X, H_F_Y))
        degrees['F-F per point'] = np.degrees(theta_per_example(F_F_X, F_F_Y))
        degrees['H-H per point'] = np.degrees(theta_per_example(H_H_X, H_H_Y))
        degrees['F-H per point'] = np.degrees(theta_per_example(F_H_X, F_H_Y))
    theta_optimal = procrustes_2d(H_F_X, H_F_Y)
    degrees['H-F'] = np.degrees(float(theta_optimal))
    theta_optimal = procrustes_2d(F_F_X, F_F_Y)
    degrees['F-F'] = np.degrees(float(theta_optimal))
    theta_optimal = procrustes_2d(H_H_X, H_H_Y)
    degrees['H-H'] = np.degrees(float(theta_optimal))
    theta_optimal = procrustes_2d(F_H_X, F_H_Y)
    degrees['F-H'] = np.degrees(float(theta_optimal))
    basis_1 = factual_mean_vector

    # Second basis vector: orthogonal to first, in the plane of both mean vectors
    basis_2 = hall_mean_vector - np.dot(hall_mean_vector, basis_1) * basis_1
    basis_2 = basis_2 / np.linalg.norm(basis_2)

    # Project the mean vectors onto 2D
    f_2d = np.array([1, 0])  # factual is along x-axis
    h_2d = np.array([np.dot(hall_mean_vector, basis_1), np.dot(hall_mean_vector, basis_2)])
    theta_optimal = procrustes_2d(f_2d.reshape(1, 2), h_2d.reshape(1, 2))
    degrees['theta_ref'] = np.degrees(float(theta_optimal))
    return degrees


def compute_all_consistencies(observations, split_models=None, ordered="False", degrees_per_file=None, two_topics=False,
                              title_addition=""):
    """Scatter plot correlating Geometric Consistency with Trace."""
    c_geo_all = []
    c_sm_all = []
    for s in split_models:
        c_geos = []
        c_sms = []
        for file, obs in observations.items():
            if not any([s_i in file for s_i in s]) or ordered not in file:
                continue
            c_geo_seeds = []
            for i in ["degrees_7", "degrees_21", "degrees_42"]:

                if degrees_per_file is None:
                    degrees = obs[i]
                else:
                    degrees = degrees_per_file[file][i]
                c_sm = (obs['2-2']["F-F"] + obs['2-2']["H-H"])
                c_geo_seeds.append(degrees["theta_ref"])
            c_sms.append(c_sm)
            c_geos.append(np.mean(c_geo_seeds))

        spearman_corr, sp_value = spearmanr(c_geos, c_sms)
        assert len(c_geos) == 6
        assert len(c_geos) == len(c_sms)
        c_geo_all.extend(c_geos)
        c_sm_all.extend(c_sms)
    spearman_corr, sp_value = spearmanr(c_geo_all, c_sm_all)
    print(f"Results for all models combined")

    print("P-value:", round(sp_value, 4))
    print("Spearman Correlation:", round(spearman_corr, 4))


    plt.savefig(
        f'plots/consistency_scatter_ordered_{ordered}_85{"_two_topics" if two_topics else ""}{title_addition}.pdf',
        format='pdf', dpi=300
    )
    plt.close()

    return spearman_corr, sp_value


def steps_back_correlation(observations, max_steps_back=3):
    """Bar chart showing the effect of history length (k) on hallucination probability."""
    observations = {k: v for k, v in observations.items() if "True" in k}
    delta_results = {}
    models = set(["_".join(file.split("_")[0:2]) for file in observations.keys()])
    for file, obs in observations.items():
        if not any([model in file for model in models]) or "True" not in file:
            continue
        results = obs["hidden_states"]
        degree_middle_layers = []
        for r in results:
            inner_list = []
            for a in r:
                inner_list.append(a[1])
            degree_middle_layers.append(inner_list)
        file_deltas = calculate_hallucination_deltas(degree_middle_layers, max_steps_back)
        delta_results[file] = file_deltas
    plot_steps_back_correlation(observations, delta_results, models, max_steps_back)


def plot_steps_back_correlation(observations, delta_results, models, max_steps_back=3, title_addition=""):
    dataset_ordered = ["natural_100", "triviaqa_100", "sorry_100", "do_not", "sycophancy_100", "sycophancy_negative"]
    color_blind_palette = ["#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7", "#F0E442"]
    plt.figure(figsize=(12, 8))

    n_datasets = len(dataset_ordered)
    total_bars = n_datasets + 1  # +1 for the Average bar
    bar_width = 0.8 / total_bars  # Total width of 0.8 divided among datasets
    x = np.arange(1, max_steps_back + 1)  # Base x positions
    all_deltas_per_step = {k: [] for k in range(1, max_steps_back + 1)}
    for i, ds in enumerate(dataset_ordered):
        deltas_per_step = {k: [] for k in range(1, max_steps_back + 1)}
        for model in models:
            start_key = f"{model}_{ds}"
            file_key = [file for file in observations.keys() if (file.startswith(start_key))][0]
            file_deltas = delta_results[file_key]
            for k in range(1, max_steps_back + 1):
                deltas_per_step[k].append(file_deltas[k])
                all_deltas_per_step[k].append(file_deltas[k])
        avg_deltas = [np.mean(deltas_per_step[k]) for k in range(1, max_steps_back + 1)]
        std_deltas = [np.std(deltas_per_step[k]) for k in range(1, max_steps_back + 1)]
        # Offset each dataset's bars
        offset = (i - total_bars / 2 + 0.5) * bar_width
        plt.bar(x + offset, avg_deltas, width=bar_width, yerr=std_deltas, capsize=3,
                label=ds.replace("do_not", "Do-Not-Answer").replace("natural_100", "NaturalQA").replace("triviaqa_100",
                                                                                                        "TriviaQA").replace(
                    "sorry_100", "Sorry").replace("sycophancy_negative", "S-neg").replace("sycophancy",
                                                                                          "S-pos").replace("_100",
                                                                                                           "").title(),
                alpha=0.7, color=color_blind_palette[i % len(color_blind_palette)])
    # add an average line across all datasets
    avg_all = [np.mean(all_deltas_per_step[k]) for k in range(1, max_steps_back + 1)]
    std_all = [np.std(all_deltas_per_step[k]) for k in range(1, max_steps_back + 1)]
    offset_avg = (n_datasets - total_bars / 2 + 0.5) * bar_width
    plt.bar(
        x + offset_avg,
        avg_all,
        width=bar_width,
        yerr=std_all,
        capsize=3,
        label="Average",
        alpha=0.9,
        color="grey",
        hatch="//"
    )
    plt.xticks(x, fontsize=25)

    plt.ylabel('$\Delta_{k}$', fontsize=30)
    plt.xlabel('Steps Back', fontsize=30)
    plt.yticks(fontsize=25)
    plt.grid(True, linestyle='--', alpha=0.35)
    plt.legend(fontsize=25, frameon=True)
    plt.tight_layout()
    plt.savefig(
        f'plots/steps_back_correlation{title_addition}.pdf', format='pdf', dpi=300)
    plt.close()


from collections import defaultdict


def calculate_hallucination_deltas(all_results, max_steps_back=7):
    """
    Calculates the 'Stickiness Delta' for history lengths from 1 to max_steps_back.
    Delta_k = P(1 | 1...1 [k times]) - P(1 | 01...1 [k-1 times])
    """
    metrics = {}

    for k in range(1, max_steps_back + 1):
        transition_counts = defaultdict(int)
        context_counts = defaultdict(int)
        for result in all_results:
            if len(result) < k + 1:
                continue
            # Slide a window of size k+1 over the result
            for i in range(len(result) - k):
                # The history is the previous k tokens
                history = tuple(result[i: i + k])
                # The next state is the current token
                next_state = result[i + k]
                transition_counts[(history, next_state)] += 1
                context_counts[history] += 1
        hist_deep = tuple([1] * k)
        hist_recent = tuple([0] + [1] * (k - 1))

        def get_prob_hallucination(history):
            total_seen = context_counts[history]
            if total_seen == 0:
                return 0.0
            count_hallucinations = transition_counts[(history, 1)]
            return count_hallucinations / total_seen

        prob_deep = get_prob_hallucination(hist_deep)
        prob_recent = get_prob_hallucination(hist_recent)
        delta = prob_deep - prob_recent
        metrics[k] = delta
    return metrics


def degrees_cw(all_observations, all_layers=False):
    layers = [-2] if not all_layers else [0, 1, 2, 3]
    per_layer_degrees = {}
    for layer in layers:
        degrees_per_file = {}
        for file, obs in all_observations.items():
            print(f"Processing file: {file}")
            all_inner_states = obs["hidden_states"]
            degree_middle_layers = []
            for r in all_inner_states:
                inner_list = []
                for a in r:
                    inner_list.append((a[0][layer], a[1]))
                degree_middle_layers.append(inner_list)
            degrees_per_file[file] = {}
            random.seed(42)
            degrees = plot_hidden_states_through_time(degree_middle_layers, calculate_angle_per_point=True)
            degrees_per_file[file]["degrees_42"] = degrees
            random.seed(7)
            degrees = plot_hidden_states_through_time(degree_middle_layers, calculate_angle_per_point=True)
            degrees_per_file[file]["degrees_7"] = degrees
            random.seed(21)
            degrees = plot_hidden_states_through_time(degree_middle_layers, calculate_angle_per_point=True)
            degrees_per_file[file]["degrees_21"] = degrees
        per_layer_degrees[layer] = degrees_per_file
    if not all_layers:
        return per_layer_degrees[-2]
    return per_layer_degrees


def calculate_trace_and_theta_different_length(all_observations, length=10, two_topics_41=False):
    layers = [-2]  # the 85% layer
    per_layer_degrees = {}
    for layer in layers:
        degrees_per_file = {}
        for file, obs in all_observations.items():
            print(f"Processing file: {file}")
            all_inner_states = obs["hidden_states"]
            degree_middle_layers = []
            for r in all_inner_states:
                inner_list = []
                for index, a in enumerate(r[:length]):  # only take the first 'length' hidden states
                    if two_topics_41:
                        if index == 6 or (index > 6 and (index - 6) % 5 == 0):
                            inner_list.append((a[0][layer], a[1]))
                    else:
                        inner_list.append((a[0][layer], a[1]))
                degree_middle_layers.append(inner_list)
            degrees_per_file[file] = {}
            random.seed(42)
            degrees = plot_hidden_states_through_time(degree_middle_layers, calculate_angle_per_point=True)
            degrees_per_file[file]["degrees_42"] = degrees
            random.seed(7)
            degrees = plot_hidden_states_through_time(degree_middle_layers, calculate_angle_per_point=True)
            degrees_per_file[file]["degrees_7"] = degrees
            random.seed(21)
            degrees = plot_hidden_states_through_time(degree_middle_layers, calculate_angle_per_point=True)
            degrees_per_file[file]["degrees_21"] = degrees
            transitions = {"0-1": 0, "1-0": 0, "0-0": 0, "1-1": 0}

            for result in degree_middle_layers:
                cur_transitions = {"0-1": 0, "1-0": 0, "0-0": 0, "1-1": 0}
                for j in range(0, len(result) - 1):
                    transition = f"{result[j - 1][1]}-{result[j][1]}"
                    transitions[transition] += 1
                    cur_transitions[transition] += 1
                observed = [[cur_transitions["0-0"], cur_transitions["0-1"]],
                            [cur_transitions["1-0"], cur_transitions["1-1"]]]

                assert sum([sum(row) for row in observed]) == len(
                    result) - 1, f"Expected sum of observations to be {len(result) - 2}, but got {sum([sum(row) for row in observed])}"

            final_2_2 = {"F-H": transitions["0-1"] / (transitions["0-1"] + transitions["0-0"]) if (transitions["0-1"] +
                                                                                                   transitions[
                                                                                                       "0-0"]) > 0 else 0,
                         "H-F": transitions["1-0"] / (transitions["1-0"] + transitions["1-1"]) if (transitions["1-0"] +
                                                                                                   transitions[
                                                                                                       "1-1"]) > 0 else 0,
                         "H-H": transitions["1-1"] / (transitions["1-1"] + transitions["1-0"]) if (transitions["1-1"] +
                                                                                                   transitions[
                                                                                                       "1-0"]) > 0 else 0,
                         "F-F": transitions["0-0"] / (transitions["0-0"] + transitions["0-1"]) if (transitions["0-0"] +
                                                                                                   transitions[
                                                                                                       "0-1"]) > 0 else 0}
            print("F-H:", round(final_2_2["F-H"], 4))
            print("H-F:", round(final_2_2["H-F"], 4))
            print("H-H:", round(final_2_2["H-H"], 4))
            print("F-F:", round(final_2_2["F-F"], 4))
            degrees_per_file[file]['2-2'] = final_2_2
            degrees_per_file[file]['hidden_states'] = obs["hidden_states"]
        per_layer_degrees[layer] = degrees_per_file
    return per_layer_degrees[-2]


def correlation_per_layer(all_observations):
    """Bar chart for layer-wise correlations."""
    degrees_cw_all_layers = degrees_cw(all_observations, all_layers=True)
    layers_correlations = {}
    for layer, degrees_per_file in degrees_cw_all_layers.items():
        s, p = compute_all_consistencies(all_observations, split_models=[["gpt"], ["llama"], ["Qwen"]], ordered="True",
                                         degrees_per_file=degrees_per_file)
        layers_correlations[layer] = (s, p)

    color_blind_palette = ["#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#009E73", "#CC79A7"]
    plt.figure(figsize=(10, 6))
    layers = ["Bottom", "Middle", "Upper", "Top"]
    correlations = [layers_correlations[layer][0] for layer in layers_correlations.keys()]
    plt.bar(layers, correlations, color=color_blind_palette[:len(layers)],
            edgecolor='black', linewidth=0.8, width=0.6)
    plt.xlabel("Layer", fontsize=30)
    plt.ylabel("Correlation", fontsize=30)
    plt.yticks(fontsize=25)
    plt.xticks(rotation=30, fontsize=25)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)

    # Subtle gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig('plots/correlations_per_layer.pdf', format='pdf', dpi=300)
    plt.close()


if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")
    json_files = [f for f in os.listdir("results/") if f.endswith(".json")]
    all_observations = {}

    for json_file in json_files:
        with open(os.path.join("results/", json_file), "r") as f:
            data = json.load(f)
            all_observations[json_file] = data
    print(f"Loaded {len(all_observations)} observation files.")
    observations_false = {k: v for k, v in all_observations.items() if "False" in k}
    observations_true = {k: v for k, v in all_observations.items() if "True" in k and "two_topics" not in k}
    observations_true_two_topics = {k: v for k, v in all_observations.items() if
                                    "True" in k and "two_topics" in k and "41" not in k}
    observations_true_two_topics41 = {k: v for k, v in all_observations.items() if "True" in k and "two_topics41" in k}

    observations_per_file_true_two_topics = calculate_trace_and_theta_different_length(observations_true_two_topics,
                                                                                       length=20)

    twenty_length_degrees = calculate_trace_and_theta_different_length(observations_true, length=20)

    observations_true = twenty_length_degrees

    observations_false = calculate_trace_and_theta_different_length(observations_false, length=20)

    plot_lambda(observations_true, split_models=[["gpt"], ["llama"], ["Qwen"]], ordered="True")

    steps_back_correlation(observations_true, max_steps_back=3)

    correlation_per_layer(observations_true)

    print("Info two topics data length 10")
    ten_length_degrees = calculate_trace_and_theta_different_length(observations_true, length=10)
    print("Info two topics data length 5")
    five_length_degrees = calculate_trace_and_theta_different_length(observations_true, length=5)
    print("Info two topics data length 15")

    fifteen_length_degrees = calculate_trace_and_theta_different_length(observations_true, length=15)
    print("Info two topics 4,1 data length 20")

    two_topics_4_1_degrees = calculate_trace_and_theta_different_length(observations_true_two_topics41, length=20,
                                                                        two_topics_41=True)

    print("results for ordered = True")
    results = compute_all_consistencies(observations_true, split_models=[["llama"], ["gpt"], ["Qwen"]], ordered="True",
                                        degrees_per_file=observations_true)

    print("results for ordered = False")
    results = compute_all_consistencies(observations_false, split_models=[["llama"], ["gpt"], ["Qwen"]],
                                        ordered="False",
                                        degrees_per_file=observations_false)

    print("results for two topics")
    results = compute_all_consistencies(observations_per_file_true_two_topics,
                                        split_models=[["llama"], ["gpt"], ["Qwen"]],
                                        ordered="True",
                                        degrees_per_file=observations_per_file_true_two_topics, two_topics=True)

    print("results for two topics 4-1")
    results = compute_all_consistencies(two_topics_4_1_degrees, split_models=[["llama"], ["gpt"], ["Qwen"]],
                                        ordered="True",
                                        degrees_per_file=two_topics_4_1_degrees,
                                        title_addition=" (Two Topics 4-1)")

    print("results for length = 10")

    results = compute_all_consistencies(ten_length_degrees, split_models=[["llama"], ["gpt"], ["Qwen"]],
                                        ordered="True",
                                        degrees_per_file=ten_length_degrees, title_addition=" (Length=10)")

    print("results for length = 5")
    results = compute_all_consistencies(five_length_degrees, split_models=[["llama"], ["gpt"], ["Qwen"]],
                                        ordered="True",
                                        degrees_per_file=five_length_degrees, title_addition=" (Length=5)")
    print("results for length = 15")
    results = compute_all_consistencies(fifteen_length_degrees, split_models=[["llama"], ["gpt"], ["Qwen"]],
                                        ordered="True",
                                        degrees_per_file=fifteen_length_degrees, title_addition=" (Length=15)")
