"""
Evaluate the saved SATIS model on 1,000 completely new simulated traces.

This file is separate from training. It imports the preprocessing logic and
paths from training_model.py, but the training main() function does not run
when imported. The fitted weights are loaded from the saved .keras model.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sensor_simulator import generate_ml_dataset
from training_model import (
    ARTIFACT_DIR,
    DATA_SEED,
    MODEL_PATH,
    SPIKE_THRESHOLD,
    remove_sudden_spikes_as_missing,
)


# -----------------------------------------------------------------------------
# Evaluation settings
# -----------------------------------------------------------------------------

N_EVALUATION_DATASETS = 5000
EVALUATION_SEED = 1001  # Must differ from DATA_SEED=42 used for training.
PREDICTION_BATCH_SIZE = 32

PREPROCESSING_PATH = os.path.join(
    ARTIFACT_DIR,
    "preprocessing_parameters.npz",
)
RESULTS_DIR = "satis_evaluation_5000_traces"


def check_required_files():
    required_files = [MODEL_PATH, PREPROCESSING_PATH]
    missing_files = [path for path in required_files if not os.path.exists(path)]

    if missing_files:
        missing_text = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Training artifacts are missing:\n"
            f"{missing_text}\n\n"
            "Run training_model.py first from the same working directory."
        )

    if EVALUATION_SEED == DATA_SEED:
        raise ValueError(
            "EVALUATION_SEED must differ from the training DATA_SEED."
        )


def generate_new_evaluation_data():
    """Generate and preprocess 1,000 new traces before normalization."""
    print(
        f"\nGenerating {N_EVALUATION_DATASETS} new evaluation traces "
        f"with seed={EVALUATION_SEED}..."
    )
    _, _, _, dataframe = generate_ml_dataset(
        n_datasets=N_EVALUATION_DATASETS,
        seed=EVALUATION_SEED,
    )

    input_columns = ["signal2_noisy", "signal3_noisy", "signal4_noisy"]
    target_columns = ["global_temperature"]
    dataset_ids = np.array(sorted(dataframe["dataset_id"].unique()))

    X_sequences = []
    Y_sequences = []
    reference_time = None

    for dataset_id in dataset_ids:
        sequence_df = dataframe[
            dataframe["dataset_id"] == dataset_id
        ].sort_values("time")

        current_time = sequence_df["time"].to_numpy()
        if reference_time is None:
            reference_time = current_time
        elif len(current_time) != len(reference_time) or not np.allclose(
            current_time,
            reference_time,
        ):
            raise ValueError(
                "All evaluation traces must use the same time grid."
            )

        X_raw = sequence_df[input_columns].values
        X_cleaned = remove_sudden_spikes_as_missing(
            X_raw,
            threshold=SPIKE_THRESHOLD,
        )

        observed_mask = (~np.isnan(X_cleaned)).astype(float)
        column_means = np.nanmean(X_cleaned, axis=0)
        column_means = np.where(np.isnan(column_means), 0.0, column_means)
        X_filled = np.where(np.isnan(X_cleaned), column_means, X_cleaned)
        X_derivatives = np.diff(
            X_filled,
            axis=0,
            prepend=X_filled[:1, :],
        )
        X_model = np.concatenate(
            [X_filled, observed_mask, X_derivatives],
            axis=1,
        )

        X_sequences.append(X_model)
        Y_sequences.append(sequence_df[target_columns].values)

    X_evaluation_raw = np.stack(X_sequences, axis=0)
    Y_evaluation_true = np.stack(Y_sequences, axis=0)

    print("Raw evaluation X shape:", X_evaluation_raw.shape)
    print("True evaluation Y shape:", Y_evaluation_true.shape)
    return (
        X_evaluation_raw,
        Y_evaluation_true,
        dataset_ids,
        reference_time,
    )


def load_training_normalization():
    """Load normalization learned from training; never fit on evaluation data."""
    with np.load(PREPROCESSING_PATH, allow_pickle=False) as parameters:
        return {
            "X_mean": parameters["X_mean"],
            "X_std": parameters["X_std"],
            "Y_mean": parameters["Y_mean"],
            "Y_std": parameters["Y_std"],
            "normalized_input_channels": parameters[
                "normalized_input_channels"
            ].astype(int),
        }


def normalize_evaluation_inputs(X_evaluation_raw, normalization):
    X_evaluation = X_evaluation_raw.copy()
    channels = normalization["normalized_input_channels"]
    X_evaluation[:, :, channels] = (
        X_evaluation[:, :, channels] - normalization["X_mean"]
    ) / normalization["X_std"]
    return X_evaluation


def load_trained_model():
    print("\nLoading trained model:", MODEL_PATH)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_evaluation_data(model, X_evaluation, normalization):
    print("Predicting all new evaluation traces...")
    Y_predicted_normalized = model.predict(
        X_evaluation,
        batch_size=PREDICTION_BATCH_SIZE,
        verbose=1,
    )
    Y_predicted = (
        Y_predicted_normalized * normalization["Y_std"]
        + normalization["Y_mean"]
    )
    return Y_predicted_normalized, Y_predicted


def calculate_residual_statistics(Y_true, Y_predicted):
    """Residual convention: predicted temperature minus true temperature."""
    residuals = Y_predicted[:, :, 0] - Y_true[:, :, 0]

    pointwise_mean_residual = np.mean(residuals, axis=0)
    pointwise_std_residual = np.std(residuals, axis=0, ddof=1)
    pointwise_sem_residual = pointwise_std_residual / np.sqrt(
        residuals.shape[0]
    )
    pointwise_ci95_lower = (
        pointwise_mean_residual - 1.96 * pointwise_sem_residual
    )
    pointwise_ci95_upper = (
        pointwise_mean_residual + 1.96 * pointwise_sem_residual
    )

    pointwise_mae = np.mean(np.abs(residuals), axis=0)
    pointwise_rmse = np.sqrt(np.mean(np.square(residuals), axis=0))

    global_metrics = {
        "global_mean_residual": float(np.mean(residuals)),
        "global_residual_std": float(np.std(residuals, ddof=1)),
        "global_mae": float(np.mean(np.abs(residuals))),
        "global_rmse": float(np.sqrt(np.mean(np.square(residuals)))),
    }

    return {
        "residuals": residuals,
        "pointwise_mean_residual": pointwise_mean_residual,
        "pointwise_std_residual": pointwise_std_residual,
        "pointwise_sem_residual": pointwise_sem_residual,
        "pointwise_ci95_lower": pointwise_ci95_lower,
        "pointwise_ci95_upper": pointwise_ci95_upper,
        "pointwise_mae": pointwise_mae,
        "pointwise_rmse": pointwise_rmse,
        "global_metrics": global_metrics,
    }


def plot_mean_residual_curve(time, statistics):
    mean_residual = statistics["pointwise_mean_residual"]
    std_residual = statistics["pointwise_std_residual"]
    ci_lower = statistics["pointwise_ci95_lower"]
    ci_upper = statistics["pointwise_ci95_upper"]

    plt.figure(figsize=(12, 6))
    plt.plot(
        time,
        mean_residual,
        color="tab:red",
        linewidth=2.2,
        label="Point-wise mean residual",
    )
    # 95% CI of the mean -- this is the band that should visibly narrow
    # as N_EVALUATION_DATASETS grows, since SEM = std / sqrt(n).
    plt.fill_between(
        time,
        ci_lower,
        ci_upper,
        color="tab:red",
        alpha=0.35,
        label="95% CI of the mean residual",
    )
    # ±1 std of individual trace residuals -- a property of the error
    # distribution itself, not expected to shrink with more traces.
    plt.fill_between(
        time,
        mean_residual - std_residual,
        mean_residual + std_residual,
        color="tab:red",
        alpha=0.12,
        label="±1 point-wise standard deviation",
    )
    plt.axhline(
        0.0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Zero residual",
    )
    plt.xlabel("Time")
    plt.ylabel("Residual: predicted - true temperature")
    plt.title(
        f"Mean residual over {N_EVALUATION_DATASETS} new evaluation traces"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"01_mean_residual_curve_{N_EVALUATION_DATASETS}_traces.png",
        ),
        dpi=300,
    )
    plt.close()


def plot_pointwise_error_curves(time, statistics):
    plt.figure(figsize=(12, 6))
    plt.plot(
        time,
        statistics["pointwise_mae"],
        linewidth=2,
        label="Point-wise MAE",
    )
    plt.plot(
        time,
        statistics["pointwise_rmse"],
        linewidth=2,
        label="Point-wise RMSE",
    )
    plt.xlabel("Time")
    plt.ylabel("Temperature error")
    plt.title(
        f"Prediction error over time on {N_EVALUATION_DATASETS} new traces"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"02_pointwise_mae_rmse_{N_EVALUATION_DATASETS}_traces.png",
        ),
        dpi=300,
    )
    plt.close()


def save_results(
    time,
    dataset_ids,
    Y_true,
    Y_predicted_normalized,
    Y_predicted,
    statistics,
):
    pointwise_table = pd.DataFrame(
        {
            "time": time,
            "mean_residual": statistics["pointwise_mean_residual"],
            "std_residual": statistics["pointwise_std_residual"],
            "sem_residual": statistics["pointwise_sem_residual"],
            "ci95_lower_mean_residual": statistics["pointwise_ci95_lower"],
            "ci95_upper_mean_residual": statistics["pointwise_ci95_upper"],
            "pointwise_mae": statistics["pointwise_mae"],
            "pointwise_rmse": statistics["pointwise_rmse"],
        }
    )
    pointwise_table.to_csv(
        os.path.join(RESULTS_DIR, "pointwise_residual_statistics.csv"),
        index=False,
    )

    residuals = statistics["residuals"]
    per_trace_table = pd.DataFrame(
        {
            "evaluation_index": np.arange(len(dataset_ids)),
            "dataset_id": dataset_ids,
            "mean_residual": np.mean(residuals, axis=1),
            "residual_std": np.std(residuals, axis=1, ddof=1),
            "mae": np.mean(np.abs(residuals), axis=1),
            "rmse": np.sqrt(np.mean(np.square(residuals), axis=1)),
        }
    )
    per_trace_table.to_csv(
        os.path.join(RESULTS_DIR, "per_trace_evaluation_metrics.csv"),
        index=False,
    )

    summary = {
        "training_seed": DATA_SEED,
        "evaluation_seed": EVALUATION_SEED,
        "number_of_evaluation_traces": N_EVALUATION_DATASETS,
        "residual_definition": "predicted_temperature - true_temperature",
        **statistics["global_metrics"],
    }
    with open(
        os.path.join(RESULTS_DIR, "evaluation_summary.json"),
        "w",
    ) as file:
        json.dump(summary, file, indent=2)

    np.savez_compressed(
        os.path.join(RESULTS_DIR, "evaluation_predictions_and_residuals.npz"),
        time=time,
        dataset_ids=dataset_ids,
        Y_true=Y_true,
        Y_predicted_normalized=Y_predicted_normalized,
        Y_predicted=Y_predicted,
        residuals=residuals,
    )


def main():
    check_required_files()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    (
        X_evaluation_raw,
        Y_evaluation_true,
        dataset_ids,
        time,
    ) = generate_new_evaluation_data()

    normalization = load_training_normalization()
    X_evaluation = normalize_evaluation_inputs(
        X_evaluation_raw,
        normalization,
    )

    model = load_trained_model()
    Y_predicted_normalized, Y_predicted = predict_evaluation_data(
        model,
        X_evaluation,
        normalization,
    )
    statistics = calculate_residual_statistics(
        Y_evaluation_true,
        Y_predicted,
    )

    print("\nEvaluation metrics over all traces and time points:")
    for name, value in statistics["global_metrics"].items():
        print(f"{name}: {value:.6f}")

    plot_mean_residual_curve(time, statistics)
    plot_pointwise_error_curves(time, statistics)
    save_results(
        time=time,
        dataset_ids=dataset_ids,
        Y_true=Y_evaluation_true,
        Y_predicted_normalized=Y_predicted_normalized,
        Y_predicted=Y_predicted,
        statistics=statistics,
    )

    print("\nEvaluation finished. The model was loaded and never retrained.")
    print("Saved results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()