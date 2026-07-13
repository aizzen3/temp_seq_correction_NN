

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Dense, LSTM, Multiply, Concatenate, Softmax,
    Conv1D, Bidirectional, Dropout, LayerNormalization, Lambda, Add
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sensor_simulator import generate_ml_dataset


# ============================================================
# Settings
# ============================================================

RESULTS_DIR = "satis_conv_search_results_v3"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_DATASETS = 1000
DATA_SEED = 42

SPIKE_THRESHOLD = 5.0
SPIKE_WINDOW = 25

BATCH_SIZE = 32
SEARCH_EPOCHS = 60
FINAL_EPOCHS = 200

# compact search space
FILTER_OPTIONS = [32, 64, 128, 256]
KERNEL_OPTIONS = [3, 5, 7]
DILATION_BASE_OPTIONS = [1, 2, 3]
NUM_CONV_LAYERS_OPTIONS = [2, 3]
LSTM_UNITS_OPTIONS = [64, 128]
DROPOUT_OPTIONS = [0.1, 0.2]
LEARNING_RATE_OPTIONS = [1e-3, 3e-3]

SEARCH_SEEDS = [1, 2]        # kept small on purpose -- compact search
MAX_CONFIGS = 30
STD_WEIGHT = 1.0

# volatility-weighting settings
VOLATILITY_WINDOW = 10
WEIGHT_FLOOR = 0.25          # smooth regions still get 25% weight, not 0

np.random.seed(DATA_SEED)
tf.keras.utils.set_random_seed(DATA_SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass


# ============================================================
# Adaptive spike detection (vectorized rolling Hampel filter)
# ============================================================

def remove_sudden_spikes_as_missing(X_raw, threshold=5.0, window=25):
    df = pd.DataFrame(X_raw)
    dx = df.diff().fillna(0.0)

    med = dx.rolling(window, center=True, min_periods=1).median()
    mad = (dx - med).abs().rolling(window, center=True, min_periods=1).median() + 1e-8
    z = (dx - med).abs() / mad

    X_cleaned = X_raw.copy()
    X_cleaned[(z > threshold).values] = np.nan
    return X_cleaned


# ============================================================
# Interpolate before differentiating
# ============================================================

def interpolate_missing(X_raw_cleaned):
    X_filled = X_raw_cleaned.copy()
    n, d = X_filled.shape
    t = np.arange(n)
    for j in range(d):
        col = X_filled[:, j]
        valid = ~np.isnan(col)
        if valid.sum() >= 2:
            X_filled[:, j] = np.interp(t, t[valid], col[valid])
        elif valid.sum() == 1:
            X_filled[:, j] = col[valid][0]
        else:
            X_filled[:, j] = 0.0
    return X_filled



# NEW

def time_since_last_obs(mask):
    """mask: (T, D) with 1=observed, 0=missing. Returns log1p(steps since
    last observed value), reset to 0 whenever a real observation occurs.
    Gives the model a sense of HOW STALE a filled value is, not just
    whether it was originally missing."""
    T, D = mask.shape
    delta = np.zeros((T, D))
    for j in range(D):
        gap = 0
        for t in range(T):
            if mask[t, j] == 1:
                gap = 0
            else:
                gap += 1
            delta[t, j] = gap
    return np.log1p(delta)


# ============================================================
# NEW: volatility-based per-timestep sample weights
# ============================================================

def volatility_sample_weights(y_seq, window=VOLATILITY_WINDOW, floor=WEIGHT_FLOOR):
    """y_seq: (T,) true target for one sequence. Returns (T,) weights,
    higher where the local signal is more volatile, so training loss
    isn't dominated by the easy smooth majority of the sequence."""
    s = pd.Series(y_seq).rolling(window, center=True, min_periods=1).std().fillna(0.0).values
    s_max = s.max() + 1e-8
    return floor + (1.0 - floor) * (s / s_max)


# ============================================================
# Dataset preparation
# ============================================================

def prepare_dataset():
    print("\nGenerating dataset...")

    _, _, _, df = generate_ml_dataset(n_datasets=N_DATASETS, seed=DATA_SEED)

    input_columns = ["signal2_noisy", "signal3_noisy", "signal4_noisy"]
    target_columns = ["global_temperature"]

    dataset_ids = np.array(sorted(df["dataset_id"].unique()))

    X_list, Y_list, diag_list = [], [], []

    for dataset_id in dataset_ids:
        df_i = df[df["dataset_id"] == dataset_id].sort_values("time")
        X_raw = df_i[input_columns].values

        X_raw_cleaned = remove_sudden_spikes_as_missing(
            X_raw, threshold=SPIKE_THRESHOLD, window=SPIKE_WINDOW
        )
        mask = (~np.isnan(X_raw_cleaned)).astype(float)
        X_filled = interpolate_missing(X_raw_cleaned)
        X_diff = np.diff(X_filled, axis=0, prepend=X_filled[:1, :])
        delta_t = time_since_last_obs(mask)

        # channels: values(3) + mask(3) + derivative(3) + time-since-obs(3) = 12
        X_model = np.concatenate([X_filled, mask, X_diff, delta_t], axis=1)

        Y_raw = df_i[target_columns].values

        X_list.append(X_model)
        Y_list.append(Y_raw)
        diag_list.append({
            "missing_fraction": 1.0 - mask.mean(),
            "volatility": float(Y_raw[:, 0].std())
        })

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    diag_df = pd.DataFrame(diag_list)

    rng = np.random.default_rng(DATA_SEED)
    indices = rng.permutation(len(X))
    X, Y = X[indices], Y[indices]
    diag_df = diag_df.iloc[indices].reset_index(drop=True)

    n_total = len(X)
    n_train = int(0.70 * n_total)
    n_val = int(0.15 * n_total)

    X_train_raw, Y_train_raw = X[:n_train], Y[:n_train]
    X_val_raw, Y_val_raw = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test_raw, Y_test_raw = X[n_train + n_val:], Y[n_train + n_val:]
    test_diag = diag_df.iloc[n_train + n_val:].reset_index(drop=True)

    # normalize values + derivatives + time-since-obs; leave masks as 0/1
    norm_channels = [0, 1, 2, 6, 7, 8, 9, 10, 11]
    X_mean = X_train_raw[:, :, norm_channels].mean(axis=(0, 1), keepdims=True)
    X_std = X_train_raw[:, :, norm_channels].std(axis=(0, 1), keepdims=True) + 1e-8

    def normalize_X(X_raw_):
        X_norm = X_raw_.copy()
        X_norm[:, :, norm_channels] = (X_norm[:, :, norm_channels] - X_mean) / X_std
        return X_norm

    X_train, X_val, X_test = normalize_X(X_train_raw), normalize_X(X_val_raw), normalize_X(X_test_raw)

    Y_mean = Y_train_raw.mean(axis=(0, 1), keepdims=True)
    Y_std = Y_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    Y_train = (Y_train_raw - Y_mean) / Y_std
    Y_val = (Y_val_raw - Y_mean) / Y_std
    Y_test = (Y_test_raw - Y_mean) / Y_std

    # NEW: volatility-based sample weights for training sequences
    W_train = np.stack([
        volatility_sample_weights(Y_train_raw[i, :, 0]) for i in range(len(Y_train_raw))
    ], axis=0)

    print("\nFinal shapes:")
    print("X_train:", X_train.shape, "Y_train:", Y_train.shape, "W_train:", W_train.shape)
    print("X_val:", X_val.shape, "X_test:", X_test.shape)

    return {
        "X_train": X_train, "Y_train": Y_train, "W_train": W_train,
        "X_val": X_val, "Y_val": Y_val,
        "X_test": X_test, "Y_test": Y_test,
        "Y_test_raw": Y_test_raw, "test_diag": test_diag,
        "Y_mean": Y_mean, "Y_std": Y_std,
    }


# ============================================================
# Persistence baseline
# ============================================================

def persistence_baseline_metrics(Y_test_raw):
    Y_true = Y_test_raw
    Y_pred = np.empty_like(Y_true)
    Y_pred[:, 0, :] = Y_true[:, 0, :]
    Y_pred[:, 1:, :] = Y_true[:, :-1, :]
    mae = mean_absolute_error(Y_true.flatten(), Y_pred.flatten())
    return mae


# ============================================================
# Model builder (dilation growth + residual conv + local-context attention)
# ============================================================

def build_satis_model(
    input_shape, conv_filters=(64, 128), kernel_size=5, dilation_base=2,
    lstm_units=64, dropout_rate=0.1, learning_rate=1e-3
):
    inputs = Input(shape=input_shape)

    sensor_values = inputs[:, :, :3]
    sensor_masks = inputs[:, :, 3:6]
    sensor_derivatives = inputs[:, :, 6:9]
    sensor_delta_t = inputs[:, :, 9:12]

    context = Conv1D(16, 5, padding="causal", activation="relu")(inputs)
    attention_logits = Dense(3)(context)
    mask_penalty = Lambda(lambda m: (1.0 - m) * (-1e9))(sensor_masks)
    attention_logits = Add()([attention_logits, mask_penalty])
    attention_weights = Softmax(axis=-1, name="sensor_attention")(attention_logits)
    weighted_sensors = Multiply()([sensor_values, attention_weights])

    x = Concatenate()([weighted_sensors, sensor_masks, sensor_derivatives, sensor_delta_t])

    for layer_idx, filters in enumerate(conv_filters):
        dilation_rate = int(dilation_base ** layer_idx)
        residual = x
        h = Conv1D(filters, kernel_size, padding="same",
                   dilation_rate=dilation_rate, activation="relu")(x)
        h = LayerNormalization()(h)
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, 1, padding="same")(residual)
        x = Add()([h, residual])

    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model



# Training one model 


def train_one_model(data, config, seed, epochs=60, batch_size=32, verbose=0):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    input_shape = (data["X_train"].shape[1], data["X_train"].shape[2])

    model = build_satis_model(
        input_shape=input_shape,
        conv_filters=config["conv_filters"],
        kernel_size=config["kernel_size"],
        dilation_base=config["dilation_base"],
        lstm_units=config["lstm_units"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=12, min_delta=0.0005,
                                restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                   min_lr=1e-6, verbose=0)

    history = model.fit(
        data["X_train"], data["Y_train"],
        sample_weight=data["W_train"],
        validation_data=(data["X_val"], data["Y_val"]),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop, reduce_lr], verbose=verbose
    )

    best_val_loss = float(np.min(history.history["val_loss"]))
    best_val_mae = float(np.min(history.history["val_mae"]))
    n_params = model.count_params()

    return model, best_val_loss, best_val_mae, n_params, history



# Compact random search 


def create_configs():
    configs = []
    rng = np.random.default_rng(DATA_SEED)
    for _ in range(MAX_CONFIGS):
        n_layers = int(rng.choice(NUM_CONV_LAYERS_OPTIONS))
        configs.append({
            "conv_filters": tuple(int(rng.choice(FILTER_OPTIONS)) for _ in range(n_layers)),
            "kernel_size": int(rng.choice(KERNEL_OPTIONS)),
            "dilation_base": int(rng.choice(DILATION_BASE_OPTIONS)),
            "lstm_units": int(rng.choice(LSTM_UNITS_OPTIONS)),
            "dropout_rate": float(rng.choice(DROPOUT_OPTIONS)),
            "learning_rate": float(rng.choice(LEARNING_RATE_OPTIONS)),
        })
    return configs


def run_hyperparameter_search(data):
    configs = create_configs()
    all_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] {config}")
        val_losses, val_maes, n_params = [], [], None

        for seed in SEARCH_SEEDS:
            _, val_loss, val_mae, n_params, _ = train_one_model(
                data, config, seed, epochs=SEARCH_EPOCHS, batch_size=BATCH_SIZE
            )
            val_losses.append(val_loss)
            val_maes.append(val_mae)

        mean_val_loss, std_val_loss = float(np.mean(val_losses)), float(np.std(val_losses))
        stability_score = mean_val_loss + STD_WEIGHT * std_val_loss

        all_results.append({
            **config,
            "mean_val_loss": mean_val_loss, "std_val_loss": std_val_loss,
            "mean_val_mae": float(np.mean(val_maes)), "std_val_mae": float(np.std(val_maes)),
            "stability_score": stability_score, "n_params": n_params
        })

        pd.DataFrame(all_results).to_csv(
            os.path.join(RESULTS_DIR, "search_partial_results.csv"), index=False
        )

    results_df = pd.DataFrame(all_results).sort_values(
        by=["stability_score", "mean_val_loss"]
    ).reset_index(drop=True)
    results_df.to_csv(os.path.join(RESULTS_DIR, "search_results.csv"), index=False)

    print("\nTop 10 configurations:")
    print(results_df.head(10))
    return results_df


# Final model training (single model, best config)


def train_final_model(data, best_config):
    print("\nTraining final model on best config...")
    model, val_loss, val_mae, n_params, history = train_one_model(
        data, best_config, seed=DATA_SEED, epochs=FINAL_EPOCHS,
        batch_size=BATCH_SIZE, verbose=1
    )
    model.save(os.path.join(RESULTS_DIR, "best_satis_conv_bilstm_model.keras"))
    return model, history


# ============================================================
# Plotting
# ============================================================

def plot_top_configs(results_df, top_n=10):
    top_df = results_df.head(top_n).copy()
    labels = [f"{r['conv_filters']}\nk={r['kernel_size']}, lstm={r['lstm_units']}"
              for _, r in top_df.iterrows()]
    x = np.arange(len(top_df))
    plt.figure(figsize=(12, 6))
    plt.errorbar(x, top_df["mean_val_loss"], yerr=top_df["std_val_loss"], fmt="o", capsize=5)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Validation loss")
    plt.title("Top configurations: mean validation loss ± std")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "01_top_configs.png"), dpi=300)
    plt.close()


def plot_loss_vs_params(results_df):
    plt.figure(figsize=(7, 5))
    plt.scatter(results_df["n_params"], results_df["mean_val_loss"])
    plt.xlabel("Number of trainable parameters")
    plt.ylabel("Mean validation loss")
    plt.title("Validation loss vs model size")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "02_val_loss_vs_model_size.png"), dpi=300)
    plt.close()


def plot_training_curve(history):
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Final model training curve")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "03_final_training_loss_curve.png"), dpi=300)
    plt.close()


def plot_prediction_examples(Y_true, Y_pred, test_diag, best_config, n_examples=5):
    """Each title now shows MAE, missing-sensor fraction, and target
    volatility for that specific example -- so it's visible on the plot
    itself why a given example is easy or hard, instead of guesswork."""
    n_examples = min(n_examples, Y_true.shape[0])
    config_str = f"filters={best_config['conv_filters']}, lstm={best_config['lstm_units']}, lr={best_config['learning_rate']}"

    for i in range(n_examples):
        example_mae = np.abs(Y_pred[i, :, 0] - Y_true[i, :, 0]).mean()
        miss_frac = test_diag.loc[i, "missing_fraction"]
        volatility = test_diag.loc[i, "volatility"]

        plt.figure(figsize=(10, 5))
        plt.plot(Y_true[i, :, 0], label="True temperature")
        plt.plot(Y_pred[i, :, 0], label="Predicted temperature")
        plt.xlabel("Time step"); plt.ylabel("Temperature"); plt.legend()
        plt.title(
            f"Example {i} | MAE={example_mae:.3f} | "
            f"missing sensor data={miss_frac:.0%} | volatility={volatility:.2f}\n"
            f"Best config: {config_str}"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"04_prediction_example_{i}.png"), dpi=300)
        plt.close()


def plot_true_vs_pred(Y_true, Y_pred):
    y_true_flat, y_pred_flat = Y_true.flatten(), Y_pred.flatten()
    if len(y_true_flat) > 10000:
        rng = np.random.default_rng(DATA_SEED)
        idx = rng.choice(len(y_true_flat), size=10000, replace=False)
        y_true_flat, y_pred_flat = y_true_flat[idx], y_pred_flat[idx]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_flat, y_pred_flat, s=5, alpha=0.4)
    lo, hi = min(y_true_flat.min(), y_pred_flat.min()), max(y_true_flat.max(), y_pred_flat.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True temperature"); plt.ylabel("Predicted temperature")
    plt.title("True vs predicted temperature")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "05_true_vs_pred_scatter.png"), dpi=300)
    plt.close()


def plot_residual_histogram(Y_true, Y_pred):
    residuals = (Y_pred - Y_true).flatten()
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Prediction error"); plt.ylabel("Count")
    plt.title("Residual histogram")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "06_residual_histogram.png"), dpi=300)
    plt.close()


def plot_error_vs_volatility(Y_true, Y_pred, test_diag):
    """Directly plots error against the two candidate causes (missing
    fraction, volatility) across ALL test examples, not just 5 -- this
    is the fastest way to confirm what's actually driving the variance
    you're seeing in the example plots."""
    per_example_mae = np.abs(Y_pred - Y_true)[:, :, 0].mean(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(test_diag["missing_fraction"], per_example_mae, s=10, alpha=0.5)
    axes[0].set_xlabel("Missing/spike sensor fraction")
    axes[0].set_ylabel("Per-example MAE")
    axes[0].set_title("Error vs missing-sensor fraction")

    axes[1].scatter(test_diag["volatility"], per_example_mae, s=10, alpha=0.5)
    axes[1].set_xlabel("True signal volatility (std)")
    axes[1].set_ylabel("Per-example MAE")
    axes[1].set_title("Error vs signal volatility")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "07_error_vs_volatility_and_missingness.png"), dpi=300)
    plt.close()

    corr_missing = np.corrcoef(test_diag["missing_fraction"], per_example_mae)[0, 1]
    corr_volatility = np.corrcoef(test_diag["volatility"], per_example_mae)[0, 1]
    print(f"\nCorrelation of per-example MAE with missing fraction: {corr_missing:.3f}")
    print(f"Correlation of per-example MAE with volatility:       {corr_volatility:.3f}")
    print("(whichever is larger in magnitude is the bigger driver of the good/bad split)")


def plot_attention_example(model, X_test, example_index=0):
    try:
        attention_model = Model(inputs=model.input,
                                 outputs=model.get_layer("sensor_attention").output)
        attention_weights = attention_model.predict(X_test[example_index:example_index + 1], verbose=0)[0]
        plt.figure(figsize=(10, 5))
        plt.plot(attention_weights[:, 0], label="Sensor 1 attention")
        plt.plot(attention_weights[:, 1], label="Sensor 2 attention")
        plt.plot(attention_weights[:, 2], label="Sensor 3 attention")
        plt.xlabel("Time step"); plt.ylabel("Attention weight"); plt.legend()
        plt.title(f"Sensor attention weights, example {example_index}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "08_sensor_attention_example.png"), dpi=300)
        plt.close()
    except Exception as e:
        print("Could not plot attention weights:", e)


# ============================================================
# Evaluation
# ============================================================

def evaluate_and_plot(final_model, history, results_df, data, best_config):
    X_test, Y_test_norm = data["X_test"], data["Y_test"]
    Y_mean, Y_std = data["Y_mean"], data["Y_std"]
    test_diag = data["test_diag"]

    print("\nPredicting on test set...")
    Y_pred_norm = final_model.predict(X_test, verbose=1)
    Y_pred = Y_pred_norm * Y_std + Y_mean
    Y_true = Y_test_norm * Y_std + Y_mean

    mse = mean_squared_error(Y_true.flatten(), Y_pred.flatten())
    mae = mean_absolute_error(Y_true.flatten(), Y_pred.flatten())
    rmse = np.sqrt(mse)
    baseline_mae = persistence_baseline_metrics(data["Y_test_raw"])
    improvement_pct = 100.0 * (baseline_mae - mae) / baseline_mae

    print(f"\nTest MSE: {mse:.5f}  RMSE: {rmse:.5f}  MAE: {mae:.5f}")
    print(f"Persistence baseline MAE: {baseline_mae:.5f}")
    print(f"Model improves on baseline by {improvement_pct:.1f}%")

    with open(os.path.join(RESULTS_DIR, "final_test_metrics.txt"), "w") as f:
        f.write(f"Test MSE: {mse}\nTest RMSE: {rmse}\nTest MAE: {mae}\n")
        f.write(f"Baseline MAE: {baseline_mae}\nImprovement: {improvement_pct:.1f}%\n")

    np.save(os.path.join(RESULTS_DIR, "Y_true_test.npy"), Y_true)
    np.save(os.path.join(RESULTS_DIR, "Y_pred_test.npy"), Y_pred)

    plot_top_configs(results_df, top_n=min(10, len(results_df)))
    plot_loss_vs_params(results_df)
    plot_training_curve(history)
    plot_prediction_examples(Y_true, Y_pred, test_diag, best_config, n_examples=5)
    plot_true_vs_pred(Y_true, Y_pred)
    plot_residual_histogram(Y_true, Y_pred)
    plot_error_vs_volatility(Y_true, Y_pred, test_diag)
    plot_attention_example(final_model, X_test, example_index=0)

    print("\nSaved all plots and results in:", RESULTS_DIR)


# ============================================================
# Main
# ============================================================

def main():
    data = prepare_dataset()
    results_df = run_hyperparameter_search(data)
    best_config = results_df.iloc[0].to_dict()

    print("\nBest configuration selected:")
    print(best_config)

    final_model, history = train_final_model(data, best_config)
    evaluate_and_plot(final_model, history, results_df, data, best_config)


if __name__ == "__main__":
    main()