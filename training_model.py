"""
Training-only script for the SATIS Conv1D-BiLSTM model.

Run this file once. It performs the Huber-loss hyperparameter search, trains
the selected final model, and saves everything needed by a separate test
script. It deliberately makes no test predictions and creates no test plots.
"""

import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Add,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Input,
    LayerNormalization,
    LSTM,
    Multiply,
    Rescaling,
    Softmax,
)
from tensorflow.keras.models import Model

from sensor_simulator import generate_ml_dataset


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

ARTIFACT_DIR = "satis_training_artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "best_satis_conv_bilstm_huber.keras")

N_DATASETS = 1000
DATA_SEED = 42
SPIKE_THRESHOLD = 5.0

BATCH_SIZE = 32
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 200

# These are the search settings from the supplied full script.
CONV_FILTER_OPTIONS = [
    (16,), (32,), (64,), (128,), (256,),
    (16, 32), (32, 64), (64, 128), (128, 256),
    (16, 32, 64), (32, 64, 128), (64, 128, 256),
]
KERNEL_OPTIONS = [3, 5, 7]
DILATION_OPTIONS = [1, 2, 4]
LSTM_UNITS_OPTIONS = [64, 128]
DROPOUT_OPTIONS = [0.1]
LEARNING_RATE_OPTIONS = [1e-3, 3e-3]

INITIAL_CONFIG = {
    "conv_filters": (64, 128),
    "kernel_size": 5,
    "dilation_rate": 1,
    "lstm_units": 64,
    "dropout_rate": 0.1,
    "learning_rate": 1e-3,
}

SEARCH_SPACE = {
    "conv_filters": CONV_FILTER_OPTIONS,
    "kernel_size": KERNEL_OPTIONS,
    "dilation_rate": DILATION_OPTIONS,
    "lstm_units": LSTM_UNITS_OPTIONS,
    "dropout_rate": DROPOUT_OPTIONS,
    "learning_rate": LEARNING_RATE_OPTIONS,
}

MAX_COORDINATE_PASSES = 3
MIN_SCORE_IMPROVEMENT = 1e-5
SEARCH_SEEDS = [1, 2, 3]
STD_WEIGHT = 1.0

np.random.seed(DATA_SEED)
tf.keras.utils.set_random_seed(DATA_SEED)

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# -----------------------------------------------------------------------------
# Dataset preparation
# -----------------------------------------------------------------------------

def remove_sudden_spikes_as_missing(X_raw, threshold=5.0):
    X_cleaned = X_raw.copy()

    for channel in range(X_raw.shape[1]):
        values = X_raw[:, channel]
        differences = np.diff(values, prepend=values[0])
        median = np.nanmedian(differences)
        mad = np.nanmedian(np.abs(differences - median)) + 1e-8
        robust_score = np.abs((differences - median) / mad)
        X_cleaned[robust_score > threshold, channel] = np.nan

    return X_cleaned


def prepare_dataset():
    print("\nGenerating dataset...")
    _, _, _, dataframe = generate_ml_dataset(
        n_datasets=N_DATASETS,
        seed=DATA_SEED,
    )

    input_columns = ["signal2_noisy", "signal3_noisy", "signal4_noisy"]
    target_columns = ["global_temperature"]
    dataset_ids = np.array(sorted(dataframe["dataset_id"].unique()))

    X_sequences = []
    Y_sequences = []

    for dataset_id in dataset_ids:
        sequence_df = dataframe[dataframe["dataset_id"] == dataset_id].sort_values("time")
        X_raw = sequence_df[input_columns].values
        X_cleaned = remove_sudden_spikes_as_missing(
            X_raw,
            threshold=SPIKE_THRESHOLD,
        )

        observed_mask = (~np.isnan(X_cleaned)).astype(float)
        column_means = np.nanmean(X_cleaned, axis=0)
        column_means = np.where(np.isnan(column_means), 0.0, column_means)
        X_filled = np.where(np.isnan(X_cleaned), column_means, X_cleaned)
        X_derivatives = np.diff(X_filled, axis=0, prepend=X_filled[:1, :])
        X_model = np.concatenate(
            [X_filled, observed_mask, X_derivatives],
            axis=1,
        )

        X_sequences.append(X_model)
        Y_sequences.append(sequence_df[target_columns].values)

    X = np.stack(X_sequences, axis=0)
    Y = np.stack(Y_sequences, axis=0)
    print("Raw X shape:", X.shape)
    print("Raw Y shape:", Y.shape)

    rng = np.random.default_rng(DATA_SEED)
    shuffled_indices = rng.permutation(len(X))
    X = X[shuffled_indices]
    Y = Y[shuffled_indices]
    dataset_ids = dataset_ids[shuffled_indices]

    n_total = len(X)
    n_train = int(0.70 * n_total)
    n_validation = int(0.15 * n_total)
    validation_end = n_train + n_validation

    X_train_raw = X[:n_train]
    Y_train_raw = Y[:n_train]
    X_validation_raw = X[n_train:validation_end]
    Y_validation_raw = Y[n_train:validation_end]
    X_test_raw = X[validation_end:]
    Y_test_raw = Y[validation_end:]

    train_dataset_ids = dataset_ids[:n_train]
    validation_dataset_ids = dataset_ids[n_train:validation_end]
    test_dataset_ids = dataset_ids[validation_end:]

    normalized_input_channels = np.array([0, 1, 2, 6, 7, 8], dtype=int)
    X_mean = X_train_raw[:, :, normalized_input_channels].mean(
        axis=(0, 1),
        keepdims=True,
    )
    X_std = X_train_raw[:, :, normalized_input_channels].std(
        axis=(0, 1),
        keepdims=True,
    ) + 1e-8

    def normalize_X(X_raw_split):
        X_normalized = X_raw_split.copy()
        X_normalized[:, :, normalized_input_channels] = (
            X_normalized[:, :, normalized_input_channels] - X_mean
        ) / X_std
        return X_normalized

    X_train = normalize_X(X_train_raw)
    X_validation = normalize_X(X_validation_raw)
    X_test = normalize_X(X_test_raw)

    Y_mean = Y_train_raw.mean(axis=(0, 1), keepdims=True)
    Y_std = Y_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    Y_train = (Y_train_raw - Y_mean) / Y_std
    Y_validation = (Y_validation_raw - Y_mean) / Y_std
    Y_test = (Y_test_raw - Y_mean) / Y_std

    print("\nNormalized shapes:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
    print("X_validation:", X_validation.shape, "Y_validation:", Y_validation.shape)
    print("X_test:", X_test.shape, "Y_test:", Y_test.shape)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_validation": X_validation,
        "Y_validation": Y_validation,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_validation_raw": Y_validation_raw,
        "Y_test_raw": Y_test_raw,
        "X_mean": X_mean,
        "X_std": X_std,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "normalized_input_channels": normalized_input_channels,
        "train_dataset_ids": train_dataset_ids,
        "validation_dataset_ids": validation_dataset_ids,
        "test_dataset_ids": test_dataset_ids,
        "input_columns": input_columns,
        "target_columns": target_columns,
    }


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

def build_satis_model(
    input_shape,
    conv_filters=(64, 128),
    kernel_size=5,
    dilation_rate=1,
    lstm_units=64,
    dropout_rate=0.1,
    learning_rate=1e-3,
):
    inputs = Input(shape=input_shape)
    sensor_values = inputs[:, :, :3]
    sensor_masks = inputs[:, :, 3:6]
    sensor_derivatives = inputs[:, :, 6:9]

    attention_logits = Dense(3)(inputs)

    # Equivalent to (1 - mask) * -1e9, but safely serializable.
    mask_penalty = Rescaling(scale=1e9, offset=-1e9)(sensor_masks)
    attention_logits = Add()([attention_logits, mask_penalty])
    attention_weights = Softmax(axis=-1, name="sensor_attention")(attention_logits)
    weighted_sensors = Multiply()([sensor_values, attention_weights])
    satis_features = Concatenate()(
        [weighted_sensors, sensor_masks, sensor_derivatives]
    )

    x = satis_features
    for filters in conv_filters:
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            activation="relu",
        )(x)
        x = LayerNormalization()(x)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"],
    )
    return model


# -----------------------------------------------------------------------------
# Coordinate-descent search using Huber loss
# -----------------------------------------------------------------------------

def train_search_model(data, config, seed):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    model = build_satis_model(
        input_shape=(data["X_train"].shape[1], data["X_train"].shape[2]),
        conv_filters=config["conv_filters"],
        kernel_size=config["kernel_size"],
        dilation_rate=config["dilation_rate"],
        lstm_units=config["lstm_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=12,
            min_delta=0.0005,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    history = model.fit(
        data["X_train"],
        data["Y_train"],
        validation_data=(data["X_validation"], data["Y_validation"]),
        epochs=SEARCH_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )

    return {
        "best_val_loss": float(np.min(history.history["val_loss"])),
        "best_val_mae": float(np.min(history.history["val_mae"])),
        "n_params": model.count_params(),
        "val_loss_curve": [float(value) for value in history.history["val_loss"]],
    }


def config_key(config):
    return (
        tuple(config["conv_filters"]),
        int(config["kernel_size"]),
        int(config["dilation_rate"]),
        int(config["lstm_units"]),
        float(config["dropout_rate"]),
        float(config["learning_rate"]),
    )


def evaluate_config(data, config, cache, curves_cache, pass_number, search_parameter):
    key = config_key(config)
    if key in cache:
        cached_result = cache[key].copy()
        cached_result.update(
            {
                "pass_number": pass_number,
                "search_parameter": search_parameter,
                "from_cache": True,
            }
        )
        print("Using cached result:", config)
        return cached_result

    print("Testing:", config)
    val_losses = []
    val_maes = []
    seed_curves = []
    parameter_count = None

    for seed in SEARCH_SEEDS:
        seed_result = train_search_model(data=data, config=config, seed=seed)
        val_losses.append(seed_result["best_val_loss"])
        val_maes.append(seed_result["best_val_mae"])
        parameter_count = seed_result["n_params"]
        seed_curves.append((seed, seed_result["val_loss_curve"]))
        print(
            f"  Seed {seed}: val_loss={seed_result['best_val_loss']:.6f}, "
            f"val_mae={seed_result['best_val_mae']:.6f}"
        )

    curves_cache[key] = {
        "config": config.copy(),
        "seed_curves": seed_curves,
    }

    result = {
        "conv_filters": tuple(config["conv_filters"]),
        "num_conv_layers": len(config["conv_filters"]),
        "kernel_size": int(config["kernel_size"]),
        "dilation_rate": int(config["dilation_rate"]),
        "lstm_units": int(config["lstm_units"]),
        "dropout_rate": float(config["dropout_rate"]),
        "learning_rate": float(config["learning_rate"]),
        "mean_val_loss": float(np.mean(val_losses)),
        "std_val_loss": float(np.std(val_losses)),
        "mean_val_mae": float(np.mean(val_maes)),
        "std_val_mae": float(np.std(val_maes)),
        "n_params": parameter_count,
        "pass_number": pass_number,
        "search_parameter": search_parameter,
        "from_cache": False,
    }
    result["stability_score"] = result["mean_val_loss"] + STD_WEIGHT * result["std_val_loss"]
    cache[key] = result.copy()
    return result


def save_search_progress(all_results, trajectory, current_config):
    pd.DataFrame(all_results).to_csv(
        os.path.join(ARTIFACT_DIR, "coordinate_search_all_trials.csv"),
        index=False,
    )
    pd.DataFrame(trajectory).to_csv(
        os.path.join(ARTIFACT_DIR, "coordinate_search_trajectory.csv"),
        index=False,
    )
    pd.DataFrame(
        [{**current_config, "num_conv_layers": len(current_config["conv_filters"])}]
    ).to_csv(
        os.path.join(ARTIFACT_DIR, "coordinate_search_current_best.csv"),
        index=False,
    )


def save_search_curves(curves_cache):
    rows = []
    for config_number, entry in enumerate(curves_cache.values()):
        config = entry["config"]
        for seed, curve in entry["seed_curves"]:
            for epoch, val_loss in enumerate(curve, start=1):
                rows.append(
                    {
                        "config_number": config_number,
                        "conv_filters": str(tuple(config["conv_filters"])),
                        "kernel_size": config["kernel_size"],
                        "dilation_rate": config["dilation_rate"],
                        "lstm_units": config["lstm_units"],
                        "dropout_rate": config["dropout_rate"],
                        "learning_rate": config["learning_rate"],
                        "seed": seed,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    }
                )

    pd.DataFrame(rows).to_csv(
        os.path.join(ARTIFACT_DIR, "search_validation_loss_curves.csv"),
        index=False,
    )


def run_hyperparameter_search(data):
    current_config = INITIAL_CONFIG.copy()
    cache = {}
    curves_cache = {}
    all_results = []
    trajectory = []

    initial_result = evaluate_config(
        data,
        current_config,
        cache,
        curves_cache,
        pass_number=0,
        search_parameter="initial_config",
    )
    all_results.append(initial_result)
    current_score = initial_result["stability_score"]
    trajectory.append(
        {
            "step": 0,
            "pass_number": 0,
            "search_parameter": "initial_config",
            "selected_value": str(current_config),
            "stability_score": current_score,
            **current_config,
        }
    )

    step = 0
    for pass_number in range(1, MAX_COORDINATE_PASSES + 1):
        print("\n" + "=" * 70)
        print(f"COORDINATE-DESCENT PASS {pass_number}/{MAX_COORDINATE_PASSES}")
        print("=" * 70)
        pass_start_score = current_score

        for parameter, candidate_values in SEARCH_SPACE.items():
            print(f"\nOptimizing only: {parameter}")
            print("Current configuration:", current_config)
            coordinate_results = []

            for candidate_value in candidate_values:
                candidate_config = current_config.copy()
                candidate_config[parameter] = candidate_value
                result = evaluate_config(
                    data,
                    candidate_config,
                    cache,
                    curves_cache,
                    pass_number,
                    parameter,
                )
                all_results.append(result)
                coordinate_results.append((result, candidate_config))

            best_result, best_candidate = min(
                coordinate_results,
                key=lambda item: (
                    item[0]["stability_score"],
                    item[0]["mean_val_loss"],
                    item[0]["std_val_loss"],
                ),
            )

            previous_value = current_config[parameter]
            previous_score = current_score
            candidate_score = best_result["stability_score"]

            if candidate_score < current_score - MIN_SCORE_IMPROVEMENT:
                current_config = best_candidate.copy()
                current_score = candidate_score
                decision = "updated"
            else:
                decision = "kept_previous"

            step += 1
            trajectory.append(
                {
                    "step": step,
                    "pass_number": pass_number,
                    "search_parameter": parameter,
                    "previous_value": str(previous_value),
                    "selected_value": str(current_config[parameter]),
                    "previous_score": previous_score,
                    "stability_score": current_score,
                    "decision": decision,
                    **current_config,
                }
            )
            print(f"Decision: {decision}; current score: {current_score:.6f}")
            save_search_progress(all_results, trajectory, current_config)

        pass_improvement = pass_start_score - current_score
        print(f"Pass improvement: {pass_improvement:.8f}")
        if pass_improvement < MIN_SCORE_IMPROVEMENT:
            print("No meaningful improvement in this pass. Stopping early.")
            break

    search_results = pd.DataFrame(list(cache.values())).sort_values(
        by=["stability_score", "mean_val_loss", "std_val_loss"]
    ).reset_index(drop=True)
    search_results.to_csv(
        os.path.join(ARTIFACT_DIR, "hyperparameter_search_results.csv"),
        index=False,
    )
    save_search_progress(all_results, trajectory, current_config)
    save_search_curves(curves_cache)

    print("\nSelected configuration:")
    print(current_config)
    print(f"Selected stability score: {current_score:.6f}")
    return current_config


# -----------------------------------------------------------------------------
# Final training and artifact saving
# -----------------------------------------------------------------------------

def train_final_model(data, best_config):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(DATA_SEED)

    model = build_satis_model(
        input_shape=(data["X_train"].shape[1], data["X_train"].shape[2]),
        conv_filters=tuple(best_config["conv_filters"]),
        kernel_size=int(best_config["kernel_size"]),
        dilation_rate=int(best_config["dilation_rate"]),
        lstm_units=int(best_config["lstm_units"]),
        dropout_rate=float(best_config["dropout_rate"]),
        learning_rate=float(best_config["learning_rate"]),
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=20,
            min_delta=0.0005,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        data["X_train"],
        data["Y_train"],
        validation_data=(data["X_validation"], data["Y_validation"]),
        epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def json_ready_config(config):
    return {
        "conv_filters": list(config["conv_filters"]),
        "kernel_size": int(config["kernel_size"]),
        "dilation_rate": int(config["dilation_rate"]),
        "lstm_units": int(config["lstm_units"]),
        "dropout_rate": float(config["dropout_rate"]),
        "learning_rate": float(config["learning_rate"]),
        "loss": "huber",
        "huber_delta": 1.0,
    }


def save_training_artifacts(data, model, history, best_config):
    model.save(MODEL_PATH)

    pd.DataFrame(history.history).to_csv(
        os.path.join(ARTIFACT_DIR, "final_training_history.csv"),
        index_label="epoch_zero_based",
    )

    with open(os.path.join(ARTIFACT_DIR, "best_config.json"), "w") as file:
        json.dump(json_ready_config(best_config), file, indent=2)

    np.savez_compressed(
        os.path.join(ARTIFACT_DIR, "preprocessing_parameters.npz"),
        X_mean=data["X_mean"],
        X_std=data["X_std"],
        Y_mean=data["Y_mean"],
        Y_std=data["Y_std"],
        normalized_input_channels=data["normalized_input_channels"],
    )

    np.savez_compressed(
        os.path.join(ARTIFACT_DIR, "validation_data.npz"),
        X_validation=data["X_validation"],
        Y_validation_normalized=data["Y_validation"],
        Y_validation_raw=data["Y_validation_raw"],
        validation_dataset_ids=data["validation_dataset_ids"],
    )

    # Saved but never predicted here. The future test script will load this.
    np.savez_compressed(
        os.path.join(ARTIFACT_DIR, "test_data.npz"),
        X_test=data["X_test"],
        Y_test_normalized=data["Y_test"],
        Y_test_raw=data["Y_test_raw"],
        test_dataset_ids=data["test_dataset_ids"],
    )

    manifest = {
        "model_path": MODEL_PATH,
        "training_loss": "Huber(delta=1.0)",
        "data_seed": DATA_SEED,
        "search_seeds": SEARCH_SEEDS,
        "n_datasets": N_DATASETS,
        "train_shape": list(data["X_train"].shape),
        "validation_shape": list(data["X_validation"].shape),
        "test_shape": list(data["X_test"].shape),
        "input_columns": data["input_columns"],
        "target_columns": data["target_columns"],
        "test_set_used_for_training_or_selection": False,
    }
    with open(os.path.join(ARTIFACT_DIR, "training_manifest.json"), "w") as file:
        json.dump(manifest, file, indent=2)


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    data = prepare_dataset()
    best_config = run_hyperparameter_search(data)

    print("\n" + "=" * 70)
    print("FINAL HUBER MODEL TRAINING")
    print("=" * 70)
    print("Configuration:", best_config)

    final_model, final_history = train_final_model(data, best_config)
    save_training_artifacts(
        data=data,
        model=final_model,
        history=final_history,
        best_config=best_config,
    )

    print("\nTraining finished. No test predictions were made.")
    print("Saved training artifacts in:", ARTIFACT_DIR)
    print("Saved model:", MODEL_PATH)


if __name__ == "__main__":
    main()