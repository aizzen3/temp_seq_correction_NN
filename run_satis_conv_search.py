
# SATIS + Conv1D + BiLSTM hyperparameter search script


import os
import itertools
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


RESULTS_DIR = "satis_conv_search_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_DATASETS = 1000
DATA_SEED = 42

SPIKE_THRESHOLD = 5.0

BATCH_SIZE = 32
SEARCH_EPOCHS = 80
FINAL_EPOCHS = 200

# Hyperparameter search settings
FILTER_OPTIONS = [16, 32, 64, 128, 256]
KERNEL_OPTIONS = [3, 5, 7]
DILATION_OPTIONS = [1, 2, 4]
NUM_CONV_LAYERS_OPTIONS = [1, 2, 3]


LSTM_UNITS_OPTIONS = [64, 128]

DROPOUT_OPTIONS = [0.1]


LEARNING_RATE_OPTIONS = [1e-3, 3e-3]

# repeated seeds for standard deviation
SEARCH_SEEDS = [1, 2, 3]


MAX_CONFIGS = 80

STD_WEIGHT = 1

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

    return best_val_loss, best_val_mae, n_params



# Hyperparameter configurations


def create_configs():
    configs = []

    for n_layers in NUM_CONV_LAYERS_OPTIONS:

        for filters in itertools.product(
            FILTER_OPTIONS,
            repeat=n_layers
        ):

            for kernel_size in KERNEL_OPTIONS:

                for dilation_rate in DILATION_OPTIONS:

                    for lstm_units in LSTM_UNITS_OPTIONS:

                        for dropout_rate in DROPOUT_OPTIONS:

                            for learning_rate in LEARNING_RATE_OPTIONS:

                                configs.append({
                                    "conv_filters": tuple(filters),
                                    "num_conv_layers": n_layers,
                                    "kernel_size": kernel_size,
                                    "dilation_rate": dilation_rate,
                                    "lstm_units": lstm_units,
                                    "dropout_rate": dropout_rate,
                                    "learning_rate": learning_rate
                                })

    print("\nTotal possible configurations:", len(configs))

    if MAX_CONFIGS is not None and MAX_CONFIGS < len(configs):
        rng = np.random.default_rng(DATA_SEED)
        chosen_indices = rng.choice(
            len(configs),
            size=MAX_CONFIGS,
            replace=False
        )
        configs = [configs[i] for i in chosen_indices]

        print("Randomly selected configurations:", len(configs))
    else:
        print("Running all configurations.")

    return configs



# Hyperparameter search
def run_hyperparameter_search(data):
    configs = create_configs()

    all_results = []

    for i, config in enumerate(configs):

        print("\n====================================================")
        print(f"Testing config {i + 1}/{len(configs)}")
        print(config)
        print("====================================================")

        val_losses = []
        val_maes = []
        param_count = None

        for seed in SEARCH_SEEDS:

            val_loss, val_mae, n_params = train_one_model(
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

            print(
                f"Seed {seed}: "
                f"val_loss={val_loss:.6f}, "
                f"val_mae={val_mae:.6f}"
            )

        mean_val_loss = float(np.mean(val_losses))
        std_val_loss = float(np.std(val_losses))

        mean_val_mae = float(np.mean(val_maes))
        std_val_mae = float(np.std(val_maes))

        stability_score = mean_val_loss + STD_WEIGHT * std_val_loss

        result = {
            "conv_filters": config["conv_filters"],
            "num_conv_layers": config["num_conv_layers"],
            "kernel_size": config["kernel_size"],
            "dilation_rate": config["dilation_rate"],
            "lstm_units": config["lstm_units"],
            "dropout_rate": config["dropout_rate"],
            "learning_rate": config["learning_rate"],
            "mean_val_loss": mean_val_loss,
            "std_val_loss": std_val_loss,
            "mean_val_mae": mean_val_mae,
            "std_val_mae": std_val_mae,
            "stability_score": stability_score,
            "n_params": param_count
        }

        all_results.append(result)

        # save temporary results every time
        temp_df = pd.DataFrame(all_results)
        temp_df.to_csv(
            os.path.join(
                RESULTS_DIR,
                "hyperparameter_search_partial_results.csv"
            ),
            index=False
        )

    results_df = pd.DataFrame(all_results)

    results_df = results_df.sort_values(
        by=["stability_score", "mean_val_loss", "std_val_loss"]
    ).reset_index(drop=True)

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            "hyperparameter_search_results.csv"
        ),
        index=False
    )

    print("\nTop 10 configurations:")
    print(results_df.head(10))

    return results_df



# Final model training

def train_final_model(data, best_config):

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(DATA_SEED)

    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_val = data["X_val"]
    Y_val = data["Y_val"]

    input_shape = (X_train.shape[1], X_train.shape[2])

    final_model = build_satis_model(
        input_shape=input_shape,
        conv_filters=tuple(best_config["conv_filters"]),
        kernel_size=int(best_config["kernel_size"]),
        dilation_rate=int(best_config["dilation_rate"]),
        lstm_units=int(best_config["lstm_units"]),
        dropout_rate=float(best_config["dropout_rate"]),
        learning_rate=float(best_config["learning_rate"])
    )

    final_model.summary()

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
        verbose=1
    )

    history = final_model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    final_model.save(
        os.path.join(
            RESULTS_DIR,
            "best_satis_conv_bilstm_model.keras"
        )
    )

    return final_model, history


# ============================================================
# Plotting functions
# ============================================================

def plot_top_configs(results_df, top_n=10):
    top_df = results_df.head(top_n).copy()

    labels = [
        f"{row['conv_filters']}\nk={row['kernel_size']}, d={row['dilation_rate']}"
        for _, row in top_df.iterrows()
    ]

    x = np.arange(len(top_df))

    plt.figure(figsize=(12, 6))
    plt.errorbar(
        x,
        top_df["mean_val_loss"],
        yerr=top_df["std_val_loss"],
        fmt="o",
        capsize=5
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Validation loss")
    plt.title("Top Conv1D configurations: mean validation loss ± std")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "01_top_configs_mean_std_val_loss.png"),
        dpi=300
    )
    plt.close()


def plot_loss_vs_params(results_df):
    plt.figure(figsize=(7, 5))
    plt.scatter(
        results_df["n_params"],
        results_df["mean_val_loss"]
    )
    plt.xlabel("Number of trainable parameters")
    plt.ylabel("Mean validation loss")
    plt.title("Validation loss vs model size")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "02_val_loss_vs_model_size.png"),
        dpi=300
    )
    plt.close()


def plot_training_curve(history):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final model training curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "03_final_training_loss_curve.png"),
        dpi=300
    )
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(history.history["mae"], label="train MAE")
    plt.plot(history.history["val_mae"], label="val MAE")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.title("Final model MAE curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "04_final_training_mae_curve.png"),
        dpi=300
    )
    plt.close()


def plot_prediction_examples(Y_true, Y_pred, best_config, n_examples=5):
    """
    CHANGE 3: title now explicitly states this is the best configuration
    found by the search, with its key hyperparameters shown, so it is
    unambiguous that examples 0-4 come from that model (not some other
    config).
    """
    n_examples = min(n_examples, Y_true.shape[0])

    config_str = (
        f"filters={tuple(best_config['conv_filters'])}, "
        f"kernel={best_config['kernel_size']}, "
        f"dilation={best_config['dilation_rate']}, "
        f"lstm_units={best_config['lstm_units']}, "
        f"lr={best_config['learning_rate']}"
    )

    print(f"\nPlotting prediction examples using BEST config: {config_str}")

    for i in range(n_examples):

        plt.figure(figsize=(10, 5))
        plt.plot(
            Y_true[i, :, 0],
            label="True temperature"
        )
        plt.plot(
            Y_pred[i, :, 0],
            label="Predicted temperature"
        )
        plt.xlabel("Time step")
        plt.ylabel("Temperature")
        plt.title(f"Prediction example {i}\nBest config: {config_str}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"05_prediction_example_{i}.png"
            ),
            dpi=300
        )
        plt.close()


def plot_true_vs_pred(Y_true, Y_pred):
    y_true_flat = Y_true.flatten()
    y_pred_flat = Y_pred.flatten()

    max_points = 10000

    if len(y_true_flat) > max_points:
        rng = np.random.default_rng(DATA_SEED)
        idx = rng.choice(
            len(y_true_flat),
            size=max_points,
            replace=False
        )
        y_true_flat = y_true_flat[idx]
        y_pred_flat = y_pred_flat[idx]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true_flat,
        y_pred_flat,
        s=5,
        alpha=0.4
    )

    min_val = min(y_true_flat.min(), y_pred_flat.min())
    max_val = max(y_true_flat.max(), y_pred_flat.max())

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel("True temperature")
    plt.ylabel("Predicted temperature")
    plt.title("True vs predicted temperature")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "06_true_vs_pred_scatter.png"),
        dpi=300
    )
    plt.close()


def plot_residual_histogram(Y_true, Y_pred):
    residuals = (Y_pred - Y_true).flatten()

    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Prediction error")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "07_residual_histogram.png"),
        dpi=300
    )
    plt.close()


def plot_error_over_time(Y_true, Y_pred):
    abs_error = np.abs(Y_pred - Y_true)[:, :, 0]

    mean_error = abs_error.mean(axis=0)
    std_error = abs_error.std(axis=0)

    t = np.arange(len(mean_error))

    plt.figure(figsize=(10, 5))
    plt.plot(
        t,
        mean_error,
        label="Mean absolute error"
    )
    plt.fill_between(
        t,
        mean_error - std_error,
        mean_error + std_error,
        alpha=0.3,
        label="±1 std"
    )
    plt.xlabel("Time step")
    plt.ylabel("Absolute error")
    plt.title("Prediction error over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_DIR, "08_error_over_time_mean_std.png"),
        dpi=300
    )
    plt.close()


def plot_attention_example(model, X_test, example_index=0):
    try:
        attention_model = Model(
            inputs=model.input,
            outputs=model.get_layer("sensor_attention").output
        )

        attention_weights = attention_model.predict(
            X_test[example_index:example_index + 1],
            verbose=0
        )[0]

        plt.figure(figsize=(10, 5))
        plt.plot(attention_weights[:, 0], label="Sensor 1 attention")
        plt.plot(attention_weights[:, 1], label="Sensor 2 attention")
        plt.plot(attention_weights[:, 2], label="Sensor 3 attention")
        plt.xlabel("Time step")
        plt.ylabel("Attention weight")
        plt.title(f"Sensor attention weights example {example_index}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                "09_sensor_attention_example.png"
            ),
            dpi=300
        )
        plt.close()

    except Exception as e:
        print("Could not plot attention weights:", e)



# Evaluation


def evaluate_and_plot(final_model, history, results_df, data, best_config):

    X_test = data["X_test"]
    Y_test_norm = data["Y_test"]
    Y_mean = data["Y_mean"]
    Y_std = data["Y_std"]

    print("\nPredicting on test set...")

    Y_pred_norm = final_model.predict(
        X_test,
        verbose=1
    )

    # denormalize
    Y_pred = Y_pred_norm * Y_std + Y_mean
    Y_true = Y_test_norm * Y_std + Y_mean

    mse = mean_squared_error(
        Y_true.flatten(),
        Y_pred.flatten()
    )

    mae = mean_absolute_error(
        Y_true.flatten(),
        Y_pred.flatten()
    )

    rmse = np.sqrt(mse)

    print("\nTest metrics:")
    print("Test MSE:", mse)
    print("Test RMSE:", rmse)
    print("Test MAE:", mae)

    metrics_path = os.path.join(
        RESULTS_DIR,
        "final_test_metrics.txt"
    )

    with open(metrics_path, "w") as f:
        f.write(f"Test MSE: {mse}\n")
        f.write(f"Test RMSE: {rmse}\n")
        f.write(f"Test MAE: {mae}\n")

    # save predictions
    np.save(
        os.path.join(RESULTS_DIR, "Y_true_test.npy"),
        Y_true
    )

    np.save(
        os.path.join(RESULTS_DIR, "Y_pred_test.npy"),
        Y_pred
    )

    # plots
    plot_top_configs(results_df, top_n=10)
    plot_loss_vs_params(results_df)
    plot_training_curve(history)
    plot_prediction_examples(Y_true, Y_pred, best_config, n_examples=5)
    plot_true_vs_pred(Y_true, Y_pred)
    plot_residual_histogram(Y_true, Y_pred)
    plot_error_over_time(Y_true, Y_pred)
    plot_attention_example(final_model, X_test, example_index=0)

    print("\nSaved all plots and results in:", RESULTS_DIR)


# ============================================================
# Main
# ============================================================

def main():

    data = prepare_dataset()

    results_df = run_hyperparameter_search(data)

    best_config = results_df.iloc[0]

    print("\n====================================================")
    print("Best configuration selected:")
    print(best_config)
    print("====================================================")

    final_model, history = train_final_model(
        data=data,
        best_config=best_config
    )

    evaluate_and_plot(
        final_model=final_model,
        history=history,
        results_df=results_df,
        data=data,
        best_config=best_config
    )


if __name__ == "__main__":
    main()