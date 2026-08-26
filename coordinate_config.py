import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    LSTM,
    Multiply,
    Concatenate,
    Softmax,
    Conv1D,
    Bidirectional,
    Dropout,
    LayerNormalization,
    Lambda,
    Add
)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sensor_simulator import generate_ml_dataset



# Settings


RESULTS_DIR = "satis_conv_search_configuration"
os.makedirs(RESULTS_DIR, exist_ok=True)

CURVES_DIR = os.path.join(RESULTS_DIR, "config_val_loss_curves")
os.makedirs(CURVES_DIR, exist_ok=True)

PREDICTION_DIR = os.path.join(RESULTS_DIR, "top_config_predictions")
os.makedirs(PREDICTION_DIR, exist_ok=True)

N_DATASETS = 1000
DATA_SEED = 42

SPIKE_THRESHOLD = 5.0

BATCH_SIZE = 32
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 200

# How many of the top (lowest-loss) configurations -- the same ones shown
# as error bars in the "best configuration" plot -- to actually re-train
# and generate true-vs-predicted plots for, so you can see whether a lower
# validation loss really does translate into a visibly better prediction.
TOP_K_FOR_PREDICTION = 5
PREDICTION_EXAMPLE_INDEX = 0

# Coordinate-descent hyperparameter search settings
# Each candidate run changes exactly ONE entry of the current configuration.
CONV_FILTER_OPTIONS = [
    (16,), (32,), (64,), (128,), (256,),
    (16, 32), (32, 64), (64, 128), (128, 256),
    (16, 32, 64), (32, 64, 128), (64, 128, 256)
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
    "learning_rate": 1e-3
}

SEARCH_SPACE = {
    "conv_filters": CONV_FILTER_OPTIONS,
    "kernel_size": KERNEL_OPTIONS,
    "dilation_rate": DILATION_OPTIONS,
    "lstm_units": LSTM_UNITS_OPTIONS,
    "dropout_rate": DROPOUT_OPTIONS,
    "learning_rate": LEARNING_RATE_OPTIONS
}

MAX_COORDINATE_PASSES = 3
MIN_SCORE_IMPROVEMENT = 1e-5

# Each configuration is trained 3 separate times, from 3 different random
# seeds (different weight initialization + data shuffling). A single run
# can get lucky or unlucky, so we train multiple times and look at the
# mean and std of the validation loss across seeds ("stability_score")
# instead of trusting one run.
SEARCH_SEEDS = [1, 2, 3]
STD_WEIGHT = 1.0

np.random.seed(DATA_SEED)
tf.keras.utils.set_random_seed(DATA_SEED)


# Optional GPU memory growth
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



# Spike detection


def remove_sudden_spikes_as_missing(X_raw, threshold=5.0):
    """
    Detect sudden jumps using MAD on first difference.
    Spike points are replaced by NaN, so they are treated as missing.
    """
    X_cleaned = X_raw.copy()

    for j in range(X_raw.shape[1]):
        x = X_raw[:, j]

        dx = np.diff(x, prepend=x[0])

        med = np.nanmedian(dx)
        mad = np.nanmedian(np.abs(dx - med)) + 1e-8

        z = np.abs((dx - med) / mad)

        X_cleaned[z > threshold, j] = np.nan

    return X_cleaned



# Dataset preparation

def prepare_dataset():
    print("\nGenerating dataset...")

    X_raw_unused, Y_raw_unused, dataset_ids_unused, df = generate_ml_dataset(
        n_datasets=N_DATASETS,
        seed=DATA_SEED
    )

    input_columns = [
        "signal2_noisy",
        "signal3_noisy",
        "signal4_noisy"
    ]

    target_columns = [
        "global_temperature"
    ]

    dataset_ids = np.array(sorted(df["dataset_id"].unique()))

    X_list = []
    Y_list = []

    for dataset_id in dataset_ids:

        df_i = df[df["dataset_id"] == dataset_id].sort_values("time")

        X_raw = df_i[input_columns].values

        # Treat large spikes as missing
        X_raw_cleaned = remove_sudden_spikes_as_missing(
            X_raw,
            threshold=SPIKE_THRESHOLD
        )

        # mask: 1 = observed, 0 = missing
        mask = (~np.isnan(X_raw_cleaned)).astype(float)

        # fill missing values using column means
        col_means = np.nanmean(X_raw_cleaned, axis=0)

        # if a full column is NaN, replace mean with 0
        col_means = np.where(np.isnan(col_means), 0.0, col_means)

        X_filled = np.where(
            np.isnan(X_raw_cleaned),
            col_means,
            X_raw_cleaned
        )

        # derivatives of filled sensor values
        X_diff = np.diff(
            X_filled,
            axis=0,
            prepend=X_filled[:1, :]
        )

        # final input:
        # channels 0:3 = filled sensor values
        # channels 3:6 = masks
        # channels 6:9 = derivatives
        X_model = np.concatenate(
            [X_filled, mask, X_diff],
            axis=1
        )

        Y_raw = df_i[target_columns].values

        X_list.append(X_model)
        Y_list.append(Y_raw)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    print("Raw X shape:", X.shape)
    print("Raw Y shape:", Y.shape)

    # Shuffle datasets
    rng = np.random.default_rng(DATA_SEED)
    indices = rng.permutation(len(X))

    X = X[indices]
    Y = Y[indices]
    dataset_ids = dataset_ids[indices]

    # Train / validation / test split
    n_total = len(X)

    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    X_train_raw = X[:n_train]
    Y_train_raw = Y[:n_train]

    X_val_raw = X[n_train:n_train + n_val]
    Y_val_raw = Y[n_train:n_train + n_val]

    X_test_raw = X[n_train + n_val:]
    Y_test_raw = Y[n_train + n_val:]

    test_dataset_ids = dataset_ids[n_train + n_val:]


    # Normal


    norm_channels = [0, 1, 2, 6, 7, 8]

    X_mean = X_train_raw[:, :, norm_channels].mean(
        axis=(0, 1),
        keepdims=True
    )

    X_std = X_train_raw[:, :, norm_channels].std(
        axis=(0, 1),
        keepdims=True
    ) + 1e-8

    def normalize_X(X_raw):
        X_norm = X_raw.copy()
        X_norm[:, :, norm_channels] = (
            X_norm[:, :, norm_channels] - X_mean
        ) / X_std
        return X_norm

    X_train = normalize_X(X_train_raw)
    X_val = normalize_X(X_val_raw)
    X_test = normalize_X(X_test_raw)


    # Normali

    Y_mean = Y_train_raw.mean(axis=(0, 1), keepdims=True)
    Y_std = Y_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8

    Y_train = (Y_train_raw - Y_mean) / Y_std
    Y_val = (Y_val_raw - Y_mean) / Y_std
    Y_test = (Y_test_raw - Y_mean) / Y_std

    print("\nFinal normalized shapes:")
    print("X_train:", X_train.shape)
    print("Y_train:", Y_train.shape)
    print("X_val:", X_val.shape)
    print("Y_val:", Y_val.shape)
    print("X_test:", X_test.shape)
    print("Y_test:", Y_test.shape)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_test_raw": Y_test_raw,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "test_dataset_ids": test_dataset_ids
    }



# Model builder


def build_satis_model(
    input_shape,
    conv_filters=(64, 128),
    kernel_size=5,
    dilation_rate=1,
    lstm_units=64,
    dropout_rate=0.1,
    learning_rate=1e-3
):


    inputs = Input(shape=input_shape)

    # channels 0:3 = cleaned sensor values
    sensor_values = inputs[:, :, :3]

    # channels 3:6 = masks
    sensor_masks = inputs[:, :, 3:6]

    # channels 6:9 = derivatives
    sensor_derivatives = inputs[:, :, 6:9]

    # --------------------------------------------------------
    # Sensor attention
    # --------------------------------------------------------

    attention_logits = Dense(3)(inputs)

    mask_penalty = Lambda(
        lambda m: (1.0 - m) * (-1e9)
    )(sensor_masks)

    attention_logits = Add()([
        attention_logits,
        mask_penalty
    ])

    attention_weights = Softmax(
        axis=-1,
        name="sensor_attention"
    )(attention_logits)

    weighted_sensors = Multiply()([
        sensor_values,
        attention_weights
    ])

    satis_features = Concatenate()([
        weighted_sensors,
        sensor_masks,
        sensor_derivatives
    ])


    # Multiple Conv1D layers


    x = satis_features

    for filters in conv_filters:

        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            dilation_rate=dilation_rate,
            activation="relu"
        )(x)

        x = LayerNormalization()(x)


    # Temporal model


    x = Bidirectional(
        LSTM(
            lstm_units,
            return_sequences=True
        )
    )(x)

    x = Dropout(dropout_rate)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    outputs = Dense(1)(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )

    return model



# Training one model


def train_one_model(
    data,
    config,
    seed,
    epochs=80,
    batch_size=32,
    verbose=0
):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_satis_model(
        input_shape=input_shape,
        conv_filters=config["conv_filters"],
        kernel_size=config["kernel_size"],
        dilation_rate=config["dilation_rate"],
        lstm_units=config["lstm_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=12,
        min_delta=0.0005,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-6,
        verbose=0
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )

    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_mae = float(np.min(history.history["val_mae"]))

    n_params = model.count_params()

    # per-epoch validation loss, so the caller can plot a learning curve
    val_loss_curve = [float(v) for v in history.history["val_loss"]]

    return best_val_loss, best_val_mae, n_params, val_loss_curve



# Coordinate-descent hyperparameter search


def config_key(config):
    return (
        tuple(config["conv_filters"]),
        int(config["kernel_size"]),
        int(config["dilation_rate"]),
        int(config["lstm_units"]),
        float(config["dropout_rate"]),
        float(config["learning_rate"])
    )


def config_label(config):
    return (
        f"filters={tuple(config['conv_filters'])}, "
        f"k={config['kernel_size']}, d={config['dilation_rate']}, "
        f"lstm={config['lstm_units']}, drop={config['dropout_rate']}, "
        f"lr={config['learning_rate']}"
    )


def evaluate_config(data, config, cache, curves_cache, pass_number, search_parameter):
    key = config_key(config)
    if key in cache:
        result = cache[key].copy()
        result.update({
            "pass_number": pass_number,
            "search_parameter": search_parameter,
            "from_cache": True
        })
        print("Using cached result:", config)
        return result

    print("Testing:", config)
    val_losses, val_maes = [], []
    param_count = None
    seed_curves = []

    for seed in SEARCH_SEEDS:
        val_loss, val_mae, n_params, val_loss_curve = train_one_model(
            data=data,
            config=config,
            seed=seed,
            epochs=SEARCH_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        param_count = n_params
        seed_curves.append((seed, val_loss_curve))
        print(
            f"  Seed {seed}: val_loss={val_loss:.6f}, "
            f"val_mae={val_mae:.6f}"
        )

    # per-epoch validation loss curves (one per seed) for this configuration
    curves_cache[key] = {
        "config": config.copy(),
        "seed_curves": seed_curves
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
        "n_params": param_count,
        "pass_number": pass_number,
        "search_parameter": search_parameter,
        "from_cache": False
    }
    result["stability_score"] = (
        result["mean_val_loss"] + STD_WEIGHT * result["std_val_loss"]
    )
    cache[key] = result.copy()
    return result


def save_search_progress(all_results, trajectory, current_config):
    pd.DataFrame(all_results).to_csv(
        os.path.join(RESULTS_DIR, "coordinate_search_all_trials.csv"),
        index=False
    )
    pd.DataFrame(trajectory).to_csv(
        os.path.join(RESULTS_DIR, "coordinate_search_trajectory.csv"),
        index=False
    )
    pd.DataFrame([{
        **current_config,
        "num_conv_layers": len(current_config["conv_filters"])
    }]).to_csv(
        os.path.join(RESULTS_DIR, "coordinate_search_current_best.csv"),
        index=False
    )


def run_hyperparameter_search(data):
    """Optimize one hyperparameter at a time, then repeat full passes."""
    current_config = INITIAL_CONFIG.copy()
    cache = {}
    curves_cache = {}
    all_results = []
    trajectory = []

    print("\nInitial configuration:")
    print(current_config)

    initial_result = evaluate_config(
        data, current_config, cache, curves_cache, 0, "initial_config"
    )
    all_results.append(initial_result)
    current_score = initial_result["stability_score"]
    trajectory.append({
        "step": 0,
        "pass_number": 0,
        "search_parameter": "initial_config",
        "selected_value": str(current_config),
        "stability_score": current_score,
        **current_config
    })

    step = 0
    for pass_number in range(1, MAX_COORDINATE_PASSES + 1):
        print("\n" + "=" * 70)
        print(f"COORDINATE-DESCENT PASS {pass_number}/{MAX_COORDINATE_PASSES}")
        print("=" * 70)
        pass_start_score = current_score

        for parameter, candidate_values in SEARCH_SPACE.items():
            print("\n" + "-" * 70)
            print(f"Optimizing only: {parameter}")
            print("Current configuration:", current_config)
            print("-" * 70)

            coordinate_results = []
            for candidate_value in candidate_values:
                candidate_config = current_config.copy()
                candidate_config[parameter] = candidate_value
                result = evaluate_config(
                    data, candidate_config, cache, curves_cache,
                    pass_number, parameter
                )
                all_results.append(result)
                coordinate_results.append((result, candidate_config))

            best_result, best_candidate = min(
                coordinate_results,
                key=lambda item: (
                    item[0]["stability_score"],
                    item[0]["mean_val_loss"],
                    item[0]["std_val_loss"]
                )
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
            trajectory.append({
                "step": step,
                "pass_number": pass_number,
                "search_parameter": parameter,
                "previous_value": str(previous_value),
                "selected_value": str(current_config[parameter]),
                "previous_score": previous_score,
                "stability_score": current_score,
                "decision": decision,
                **current_config
            })

            print(f"Best tested {parameter}: {best_candidate[parameter]}")
            print(f"Candidate score: {candidate_score:.6f}")
            print(f"Decision: {decision}")
            print(f"Current score: {current_score:.6f}")
            save_search_progress(all_results, trajectory, current_config)

        pass_improvement = pass_start_score - current_score
        print(f"\nPass {pass_number} improvement: {pass_improvement:.8f}")
        if pass_improvement < MIN_SCORE_IMPROVEMENT:
            print("No meaningful improvement in this pass. Stopping early.")
            break

    # Sorted purely by score across every configuration that was ever
    # trained -- this is what "top configurations" should mean.
    unique_results_df = pd.DataFrame(list(cache.values())).sort_values(
        by=["stability_score", "mean_val_loss", "std_val_loss"]
    ).reset_index(drop=True)

    unique_results_df.to_csv(
        os.path.join(RESULTS_DIR, "hyperparameter_search_results.csv"),
        index=False
    )
    save_search_progress(all_results, trajectory, current_config)

    print("\nFinal coordinate-descent configuration:")
    print(current_config)
    print(f"Final stability score: {current_score:.6f}")
    print("\nTop 10 trained configurations (by validation loss):")
    print(unique_results_df.head(10))

    return unique_results_df, curves_cache, current_config


# Train + predict for one configuration

def train_and_predict(data, config, seed=DATA_SEED, epochs=FINAL_EPOCHS, verbose=0):
    """Train one model end-to-end and return it plus its test-set predictions."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]

    input_shape = (X_train.shape[1], X_train.shape[2])

    model = build_satis_model(
        input_shape=input_shape,
        conv_filters=tuple(config["conv_filters"]),
        kernel_size=int(config["kernel_size"]),
        dilation_rate=int(config["dilation_rate"]),
        lstm_units=int(config["lstm_units"]),
        dropout_rate=float(config["dropout_rate"]),
        learning_rate=float(config["learning_rate"])
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=20,
        min_delta=0.0005,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=0
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )

    best_val_loss = float(np.min(history.history["val_loss"]))

    Y_pred_norm = model.predict(data["X_test"], verbose=0)
    Y_pred = Y_pred_norm * data["Y_std"] + data["Y_mean"]

    return model, history, best_val_loss, Y_pred


# ============================================================
# Plotting functions
#   1) top-configuration error-bar plot (globally sorted by loss) --
#      this is the "many configs, each an error bar" plot
#   2) per-configuration validation-loss learning curves (with the
#      final loss number and a "BEST" marker on the selected config)
#   3) true-vs-predicted plots for the best few configurations from
#      plot (1), each labeled with its validation loss, so you can
#      see whether a lower loss really means a visibly better
#      prediction, or whether the two don't line up
# ============================================================

def plot_top_configs(unique_results_df, final_config_key, top_n=10):
    top_df = unique_results_df.head(top_n).copy()

    labels = [
        f"{row['conv_filters']}\nk={row['kernel_size']}, d={row['dilation_rate']}"
        for _, row in top_df.iterrows()
    ]

    x = np.arange(len(top_df))
    is_selected = [
        config_key(row.to_dict()) == final_config_key
        for _, row in top_df.iterrows()
    ]
    colors = ["crimson" if sel else "tab:blue" for sel in is_selected]

    plt.figure(figsize=(13, 6))
    for xi, (_, row), color in zip(x, top_df.iterrows(), colors):
        plt.errorbar(
            xi, row["mean_val_loss"], yerr=row["std_val_loss"],
            fmt="o", capsize=5, color=color, markersize=8
        )
        plt.text(
            xi, row["mean_val_loss"] + row["std_val_loss"] + 0.005,
            f"{row['mean_val_loss']:.4f}",
            ha="center", fontsize=8
        )

    for xi, sel in zip(x, is_selected):
        if sel:
            plt.text(xi, plt.ylim()[0], "SELECTED\nBY SEARCH", ha="center",
                      va="bottom", color="crimson", fontsize=8, fontweight="bold")

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Validation loss (mean ± std across seeds)")
    plt.title(
        "Top configurations by validation loss\n"
        "(red = the configuration the coordinate-descent search picked; "
        "it is not always the single lowest-loss config tried, since the "
        "search is greedy and doesn't test every combination)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "01_top_configs_mean_std_val_loss.png"),
        dpi=300
    )
    plt.close()


def plot_config_val_loss_curves(curves_cache, final_config_key):
    for i, (key, entry) in enumerate(curves_cache.items()):
        config = entry["config"]
        seed_curves = entry["seed_curves"]
        is_best = (key == final_config_key)

        per_seed_best = [min(curve) for _, curve in seed_curves]
        mean_best = float(np.mean(per_seed_best))

        plt.figure(figsize=(8, 5))
        for seed, val_loss_curve in seed_curves:
            epochs = np.arange(1, len(val_loss_curve) + 1)
            plt.plot(epochs, val_loss_curve,
                      label=f"seed {seed}")

        star = " \u2605 BEST (selected by search)" if is_best else ""
        plt.xlabel("Epoch")
        plt.ylabel("Validation loss")
        plt.title(
            f"Config {i}{star}\n{config_label(config)}\n"
            f"mean best val_loss across seeds = {mean_best:.5f}"
        )
        plt.legend(fontsize=7)
        plt.tight_layout()

        filename = f"config_{i:03d}_val_loss_curve.png"
        plt.savefig(os.path.join(CURVES_DIR, filename), dpi=300)
        plt.close()

    print(f"Saved {len(curves_cache)} per-configuration validation loss curves in: {CURVES_DIR}")


def plot_top_config_predictions(prediction_infos, Y_true, example_index=0):
    """
    True-vs-predicted temperature for each of the top-K configurations,
    each one labeled with its actual validation loss number, so you can
    directly compare "lower loss" against "better looking prediction".
    """
    for info in prediction_infos:
        rank = info["rank"]
        config = info["config"]
        val_loss = info["val_loss"]
        Y_pred = info["Y_pred"]
        is_best = info["is_best"]

        star = " \u2605 BEST (selected by search)" if is_best else ""

        plt.figure(figsize=(10, 5))
        plt.plot(Y_true[example_index, :, 0], label="True temperature")
        plt.plot(Y_pred[example_index, :, 0], label="Predicted temperature")
        plt.xlabel("Time step")
        plt.ylabel("Temperature")
        plt.title(
            f"Rank {rank + 1}{star}\n{config_label(config)}\n"
            f"validation loss = {val_loss:.5f}"
        )
        plt.legend()
        plt.tight_layout()

        filename = f"prediction_rank{rank + 1:02d}.png"
        plt.savefig(os.path.join(PREDICTION_DIR, filename), dpi=300)
        plt.close()

    print(f"Saved {len(prediction_infos)} top-config prediction plots in: {PREDICTION_DIR}")


# Evaluation


def evaluate_and_plot(
    final_model, results_df, curves_cache,
    final_config_key, data, prediction_infos, Y_true
):
    X_test = data["X_test"]

    Y_pred_best = next(p["Y_pred"] for p in prediction_infos if p["is_best"])

    mse = mean_squared_error(Y_true.flatten(), Y_pred_best.flatten())
    mae = mean_absolute_error(Y_true.flatten(), Y_pred_best.flatten())
    rmse = np.sqrt(mse)

    print("\nTest metrics (best/selected configuration):")
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test MAE:", mae)

    metrics_path = os.path.join(RESULTS_DIR, "final_test_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test MSE: {mse}\n")
        f.write(f"Test RMSE: {rmse}\n")
        f.write(f"Test MAE: {mae}\n")

    np.save(os.path.join(RESULTS_DIR, "Y_true_test.npy"), Y_true)
    np.save(os.path.join(RESULTS_DIR, "Y_pred_test.npy"), Y_pred_best)

    plot_top_configs(results_df, final_config_key, top_n=10)
    plot_config_val_loss_curves(curves_cache, final_config_key)
    plot_top_config_predictions(prediction_infos, Y_true, example_index=PREDICTION_EXAMPLE_INDEX)

    print("\nSaved all plots and results in:", RESULTS_DIR)


# ============================================================
# Main
# ============================================================

def main():

    data = prepare_dataset()

    results_df, curves_cache, best_config = run_hyperparameter_search(data)
    final_config_key = config_key(best_config)

    print("\n====================================================")
    print("Best configuration selected:")
    print(best_config)
    print("====================================================")

    Y_true = data["Y_test"] * data["Y_std"] + data["Y_mean"]

    # Re-train the top-K configurations (rank 0 = the one the search
    # selected) and generate real predictions for each, so you can see
    # whether the parameter changes actually improve the prediction.
    top_k_df = results_df.head(TOP_K_FOR_PREDICTION)

    prediction_infos = []
    final_model = None
    for rank, (_, row) in enumerate(top_k_df.iterrows()):
        config = row.to_dict()
        config["conv_filters"] = tuple(config["conv_filters"])
        is_best = (config_key(config) == final_config_key)

        print(f"\nTraining rank {rank + 1}/{TOP_K_FOR_PREDICTION} "
              f"for prediction comparison: {config_label(config)}")

        model, history, val_loss, Y_pred = train_and_predict(
            data, config,
            epochs=FINAL_EPOCHS if is_best else SEARCH_EPOCHS,
            verbose=1 if is_best else 0
        )

        if is_best:
            final_model = model
            final_model.save(
                os.path.join(RESULTS_DIR, "best_satis_conv_bilstm_model.keras")
            )

        prediction_infos.append({
            "rank": rank,
            "config": config,
            "val_loss": val_loss,
            "Y_pred": Y_pred,
            "is_best": is_best
        })

    evaluate_and_plot(
        final_model=final_model,
        results_df=results_df,
        curves_cache=curves_cache,
        final_config_key=final_config_key,
        data=data,
        prediction_infos=prediction_infos,
        Y_true=Y_true
    )


if __name__ == "__main__":
    main()