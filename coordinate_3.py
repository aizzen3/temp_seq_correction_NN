import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Multiply, Concatenate, Softmax, Conv1D, Bidirectional, Dropout, LayerNormalization, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sensor_simulator import generate_ml_dataset

RESULTS_DIR = "satis_conv_search_configuration3"; os.makedirs(RESULTS_DIR, exist_ok=True)
CURVES_DIR = os.path.join(RESULTS_DIR, "config_val_loss_curves"); os.makedirs(CURVES_DIR, exist_ok=True)
PREDICTION_DIR = os.path.join(RESULTS_DIR, "best_config_random_predictions"); os.makedirs(PREDICTION_DIR, exist_ok=True)
LOSS_COMPARISON_DIR = os.path.join(RESULTS_DIR, "loss_function_comparison"); os.makedirs(LOSS_COMPARISON_DIR, exist_ok=True)
LOSS_MODELS_DIR = os.path.join(LOSS_COMPARISON_DIR, "models"); os.makedirs(LOSS_MODELS_DIR, exist_ok=True)
N_DATASETS = 1000; DATA_SEED = 42; SPIKE_THRESHOLD = 5.0
BATCH_SIZE = 32; SEARCH_EPOCHS = 80; FINAL_EPOCHS = 200
# Four random test sequences are selected reproducibly.
# Change RANDOM_EXAMPLE_SEED if you want another group of four sequences.
N_RANDOM_PREDICTION_EXAMPLES = 4; RANDOM_EXAMPLE_SEED = 123

# Stage 1: focused architecture search. This search always uses Huber loss.
SEARCH_LOSS_NAME = "huber"
CONV_FILTER_OPTIONS = [(192,), (256,), (384,), (512,)]
KERNEL_OPTIONS = [3]
DILATION_OPTIONS = [2, 4, 6, 8]
LSTM_UNITS_OPTIONS = [96, 128, 160, 192]
DROPOUT_OPTIONS = [0.1, 0.2]
LEARNING_RATE_OPTIONS = [3e-4, 1e-3]
INITIAL_CONFIG = {"conv_filters": (256,), "kernel_size": 3, "dilation_rate": 4,
                  "lstm_units": 128, "dropout_rate": 0.1, "learning_rate": 1e-3}
SEARCH_SPACE = {"conv_filters": CONV_FILTER_OPTIONS, "kernel_size": KERNEL_OPTIONS, "dilation_rate": DILATION_OPTIONS, "lstm_units": LSTM_UNITS_OPTIONS, "dropout_rate": DROPOUT_OPTIONS, "learning_rate": LEARNING_RATE_OPTIONS}
MAX_COORDINATE_PASSES = 3; MIN_SCORE_IMPROVEMENT = 1e-5
SEARCH_SEEDS = [1, 2, 3]; STD_WEIGHT = 1.0

# Stage 2: compare these losses using the same selected architecture and seed.
LOSS_FUNCTION_OPTIONS = ["huber", "mae", "mse", "log_cosh"]
LOSS_DISPLAY_NAMES = {"huber": "Huber", "mae": "MAE", "mse": "MSE", "log_cosh": "Log-Cosh"}
np.random.seed(DATA_SEED); tf.keras.utils.set_random_seed(DATA_SEED)
try:
    for gpu in tf.config.list_physical_devices("GPU"): tf.config.experimental.set_memory_growth(gpu, True)
except Exception: pass


def remove_sudden_spikes_as_missing(X_raw, threshold=5.0):
    X_cleaned = X_raw.copy()
    for j in range(X_raw.shape[1]):
        x = X_raw[:, j]; dx = np.diff(x, prepend=x[0])
        med = np.nanmedian(dx); mad = np.nanmedian(np.abs(dx - med)) + 1e-8
        z = np.abs((dx - med) / mad)
        X_cleaned[z > threshold, j] = np.nan
    return X_cleaned


def prepare_dataset():
    print("\nGenerating dataset...")
    X_raw_unused, Y_raw_unused, dataset_ids_unused, df = generate_ml_dataset(n_datasets=N_DATASETS, seed=DATA_SEED)
    input_columns = ["signal2_noisy", "signal3_noisy", "signal4_noisy"]
    target_columns = ["global_temperature"]
    dataset_ids = np.array(sorted(df["dataset_id"].unique()))
    X_list, Y_list = [], []
    for dataset_id in dataset_ids:
        df_i = df[df["dataset_id"] == dataset_id].sort_values("time")
        X_raw = df_i[input_columns].values
        X_raw_cleaned = remove_sudden_spikes_as_missing(X_raw, threshold=SPIKE_THRESHOLD)
        mask = (~np.isnan(X_raw_cleaned)).astype(float)
        col_means = np.nanmean(X_raw_cleaned, axis=0); col_means = np.where(np.isnan(col_means), 0.0, col_means)
        X_filled = np.where(np.isnan(X_raw_cleaned), col_means, X_raw_cleaned)
        X_diff = np.diff(X_filled, axis=0, prepend=X_filled[:1, :])
        X_model = np.concatenate([X_filled, mask, X_diff], axis=1)
        Y_raw = df_i[target_columns].values
        X_list.append(X_model); Y_list.append(Y_raw)
    X = np.stack(X_list, axis=0); Y = np.stack(Y_list, axis=0)
    print("Raw X shape:", X.shape); print("Raw Y shape:", Y.shape)
    rng = np.random.default_rng(DATA_SEED); indices = rng.permutation(len(X))
    X, Y, dataset_ids = X[indices], Y[indices], dataset_ids[indices]
    n_total = len(X); n_train = int(0.70 * n_total); n_val = int(0.15 * n_total)
    X_train_raw, Y_train_raw = X[:n_train], Y[:n_train]
    X_val_raw, Y_val_raw = X[n_train:n_train + n_val], Y[n_train:n_train + n_val]
    X_test_raw, Y_test_raw = X[n_train + n_val:], Y[n_train + n_val:]
    test_dataset_ids = dataset_ids[n_train + n_val:]
    norm_channels = [0, 1, 2, 6, 7, 8]
    X_mean = X_train_raw[:, :, norm_channels].mean(axis=(0, 1), keepdims=True)
    X_std = X_train_raw[:, :, norm_channels].std(axis=(0, 1), keepdims=True) + 1e-8

    def normalize_X(X_raw):
        X_norm = X_raw.copy(); X_norm[:, :, norm_channels] = (X_norm[:, :, norm_channels] - X_mean) / X_std
        return X_norm

    X_train, X_val, X_test = normalize_X(X_train_raw), normalize_X(X_val_raw), normalize_X(X_test_raw)
    Y_mean = Y_train_raw.mean(axis=(0, 1), keepdims=True); Y_std = Y_train_raw.std(axis=(0, 1), keepdims=True) + 1e-8
    Y_train, Y_val, Y_test = (Y_train_raw - Y_mean) / Y_std, (Y_val_raw - Y_mean) / Y_std, (Y_test_raw - Y_mean) / Y_std
    print("\nFinal normalized shapes:")
    print("X_train:", X_train.shape); print("Y_train:", Y_train.shape)
    print("X_val:", X_val.shape); print("Y_val:", Y_val.shape)
    print("X_test:", X_test.shape); print("Y_test:", Y_test.shape)
    return {"X_train": X_train, "Y_train": Y_train, "X_val": X_val, "Y_val": Y_val, "X_test": X_test, "Y_test": Y_test,
            "Y_test_raw": Y_test_raw, "Y_mean": Y_mean, "Y_std": Y_std, "test_dataset_ids": test_dataset_ids}


def make_loss_function(loss_name):
    if loss_name == "huber":
        return tf.keras.losses.Huber(delta=1.0)
    if loss_name == "mae":
        return tf.keras.losses.MeanAbsoluteError()
    if loss_name == "mse":
        return tf.keras.losses.MeanSquaredError()
    if loss_name == "log_cosh":
        return tf.keras.losses.LogCosh()
    raise ValueError(f"Unknown loss function: {loss_name}")


def build_satis_model(input_shape, conv_filters=(64, 128), kernel_size=5, dilation_rate=1,
                      lstm_units=64, dropout_rate=0.1, learning_rate=1e-3,
                      loss_name="huber"):
    inputs = Input(shape=input_shape)
    sensor_values = inputs[:, :, :3]; sensor_masks = inputs[:, :, 3:6]; sensor_derivatives = inputs[:, :, 6:9]
    attention_logits = Dense(3)(inputs)
    mask_penalty = Lambda(lambda m: (1.0 - m) * (-1e9))(sensor_masks)
    attention_logits = Add()([attention_logits, mask_penalty])
    attention_weights = Softmax(axis=-1, name="sensor_attention")(attention_logits)
    weighted_sensors = Multiply()([sensor_values, attention_weights])
    satis_features = Concatenate()([weighted_sensors, sensor_masks, sensor_derivatives])
    x = satis_features
    for filters in conv_filters:
        x = Conv1D(filters=filters, kernel_size=kernel_size, padding="same", dilation_rate=dilation_rate, activation="relu")(x)
        x = LayerNormalization()(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=make_loss_function(loss_name), metrics=["mae"])
    return model


def train_one_model(data, config, seed, epochs=80, batch_size=32, verbose=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_satis_model(input_shape=input_shape, conv_filters=config["conv_filters"], kernel_size=config["kernel_size"],
                               dilation_rate=config["dilation_rate"], lstm_units=config["lstm_units"],
                               dropout_rate=config["dropout_rate"], learning_rate=config["learning_rate"],
                               loss_name=SEARCH_LOSS_NAME)
    early_stop = EarlyStopping(monitor="val_loss", patience=12, min_delta=0.0005, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=0)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size,
                         callbacks=[early_stop, reduce_lr], verbose=verbose)
    best_val_loss = float(np.min(history.history["val_loss"])); best_val_mae = float(np.min(history.history["val_mae"]))
    n_params = model.count_params()
    val_loss_curve = [float(v) for v in history.history["val_loss"]]
    return best_val_loss, best_val_mae, n_params, val_loss_curve


def config_key(config):
    return (tuple(config["conv_filters"]), int(config["kernel_size"]), int(config["dilation_rate"]),
            int(config["lstm_units"]), float(config["dropout_rate"]), float(config["learning_rate"]))


def config_label(config):
    return (f"filters={tuple(config['conv_filters'])}, k={config['kernel_size']}, d={config['dilation_rate']}, "
            f"lstm={config['lstm_units']}, drop={config['dropout_rate']}, lr={config['learning_rate']}")


def evaluate_config(data, config, cache, curves_cache, pass_number, search_parameter):
    key = config_key(config)
    if key in cache:
        result = cache[key].copy(); result.update({"pass_number": pass_number, "search_parameter": search_parameter, "from_cache": True})
        print("Using cached result:", config); return result
    print("Testing:", config)
    val_losses, val_maes, param_count, seed_curves = [], [], None, []
    for seed in SEARCH_SEEDS:
        val_loss, val_mae, n_params, val_loss_curve = train_one_model(data=data, config=config, seed=seed, epochs=SEARCH_EPOCHS, batch_size=BATCH_SIZE, verbose=0)
        val_losses.append(val_loss); val_maes.append(val_mae); param_count = n_params; seed_curves.append((seed, val_loss_curve))
        print(f"  Seed {seed}: val_loss={val_loss:.6f}, val_mae={val_mae:.6f}")
    curves_cache[key] = {"config": config.copy(), "seed_curves": seed_curves}
    result = {"conv_filters": tuple(config["conv_filters"]), "num_conv_layers": len(config["conv_filters"]), "kernel_size": int(config["kernel_size"]),
              "dilation_rate": int(config["dilation_rate"]), "lstm_units": int(config["lstm_units"]), "dropout_rate": float(config["dropout_rate"]),
              "learning_rate": float(config["learning_rate"]), "mean_val_loss": float(np.mean(val_losses)), "std_val_loss": float(np.std(val_losses)),
              "mean_val_mae": float(np.mean(val_maes)), "std_val_mae": float(np.std(val_maes)), "n_params": param_count,
              "pass_number": pass_number, "search_parameter": search_parameter, "from_cache": False}
    result["stability_score"] = result["mean_val_loss"] + STD_WEIGHT * result["std_val_loss"]
    cache[key] = result.copy()
    return result


def save_search_progress(all_results, trajectory, current_config):
    pd.DataFrame(all_results).to_csv(os.path.join(RESULTS_DIR, "coordinate_search_all_trials.csv"), index=False)
    pd.DataFrame(trajectory).to_csv(os.path.join(RESULTS_DIR, "coordinate_search_trajectory.csv"), index=False)
    pd.DataFrame([{**current_config, "num_conv_layers": len(current_config["conv_filters"])}]).to_csv(os.path.join(RESULTS_DIR, "coordinate_search_current_best.csv"), index=False)


def run_hyperparameter_search(data):
    current_config = INITIAL_CONFIG.copy(); cache, curves_cache, all_results, trajectory = {}, {}, [], []
    print("\nInitial configuration:"); print(current_config)
    initial_result = evaluate_config(data, current_config, cache, curves_cache, 0, "initial_config")
    all_results.append(initial_result); current_score = initial_result["stability_score"]
    trajectory.append({"step": 0, "pass_number": 0, "search_parameter": "initial_config", "selected_value": str(current_config), "stability_score": current_score, **current_config})
    step = 0
    for pass_number in range(1, MAX_COORDINATE_PASSES + 1):
        print("\n" + "=" * 70); print(f"COORDINATE-DESCENT PASS {pass_number}/{MAX_COORDINATE_PASSES}"); print("=" * 70)
        pass_start_score = current_score
        for parameter, candidate_values in SEARCH_SPACE.items():
            print("\n" + "-" * 70); print(f"Optimizing only: {parameter}"); print("Current configuration:", current_config); print("-" * 70)
            coordinate_results = []
            for candidate_value in candidate_values:
                candidate_config = current_config.copy(); candidate_config[parameter] = candidate_value
                result = evaluate_config(data, candidate_config, cache, curves_cache, pass_number, parameter)
                all_results.append(result); coordinate_results.append((result, candidate_config))
            best_result, best_candidate = min(coordinate_results, key=lambda item: (item[0]["stability_score"], item[0]["mean_val_loss"], item[0]["std_val_loss"]))
            previous_value, previous_score, candidate_score = current_config[parameter], current_score, best_result["stability_score"]
            if candidate_score < current_score - MIN_SCORE_IMPROVEMENT:
                current_config, current_score, decision = best_candidate.copy(), candidate_score, "updated"
            else:
                decision = "kept_previous"
            step += 1
            trajectory.append({"step": step, "pass_number": pass_number, "search_parameter": parameter, "previous_value": str(previous_value),
                                "selected_value": str(current_config[parameter]), "previous_score": previous_score, "stability_score": current_score,
                                "decision": decision, **current_config})
            print(f"Best tested {parameter}: {best_candidate[parameter]}"); print(f"Candidate score: {candidate_score:.6f}")
            print(f"Decision: {decision}"); print(f"Current score: {current_score:.6f}")
            save_search_progress(all_results, trajectory, current_config)
        pass_improvement = pass_start_score - current_score
        print(f"\nPass {pass_number} improvement: {pass_improvement:.8f}")
        if pass_improvement < MIN_SCORE_IMPROVEMENT:
            print("No meaningful improvement in this pass. Stopping early."); break
    unique_results_df = pd.DataFrame(list(cache.values())).sort_values(by=["stability_score", "mean_val_loss", "std_val_loss"]).reset_index(drop=True)
    unique_results_df.to_csv(os.path.join(RESULTS_DIR, "hyperparameter_search_results.csv"), index=False)
    save_search_progress(all_results, trajectory, current_config)
    print("\nFinal coordinate-descent configuration:"); print(current_config)
    print(f"Final stability score: {current_score:.6f}")
    print("\nTop 10 trained configurations (by validation loss):"); print(unique_results_df.head(10))
    return unique_results_df, curves_cache, current_config


def train_and_predict(data, config, seed=DATA_SEED, epochs=FINAL_EPOCHS, verbose=0,
                      loss_name="huber"):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_satis_model(input_shape=input_shape, conv_filters=tuple(config["conv_filters"]), kernel_size=int(config["kernel_size"]),
                               dilation_rate=int(config["dilation_rate"]), lstm_units=int(config["lstm_units"]),
                               dropout_rate=float(config["dropout_rate"]), learning_rate=float(config["learning_rate"]),
                               loss_name=loss_name)
    early_stop = EarlyStopping(monitor="val_loss", patience=20, min_delta=0.0005, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6, verbose=0)
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=BATCH_SIZE, callbacks=[early_stop, reduce_lr], verbose=verbose)
    best_val_loss = float(np.min(history.history["val_loss"]))
    Y_pred_norm = model.predict(data["X_test"], verbose=0); Y_pred = Y_pred_norm * data["Y_std"] + data["Y_mean"]
    return model, history, best_val_loss, Y_pred


def plot_top_configs(unique_results_df, final_config_key, top_n=10):
    top_df = unique_results_df.head(top_n).copy()
    labels = [f"{row['conv_filters']}\nk={row['kernel_size']}, d={row['dilation_rate']}" for _, row in top_df.iterrows()]
    x = np.arange(len(top_df))
    is_selected = [config_key(row.to_dict()) == final_config_key for _, row in top_df.iterrows()]
    colors = ["crimson" if sel else "tab:blue" for sel in is_selected]
    plt.figure(figsize=(13, 6))
    for xi, (_, row), color in zip(x, top_df.iterrows(), colors):
        plt.errorbar(xi, row["mean_val_loss"], yerr=row["std_val_loss"], fmt="o", capsize=5, color=color, markersize=8)
        plt.text(xi, row["mean_val_loss"] + row["std_val_loss"] + 0.005, f"{row['mean_val_loss']:.4f}", ha="center", fontsize=8)
    for xi, sel in zip(x, is_selected):
        if sel:
            plt.text(xi, plt.ylim()[0], "SELECTED\nBY SEARCH", ha="center", va="bottom", color="crimson", fontsize=8, fontweight="bold")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Validation loss (mean ± std across seeds)")
    plt.title("Top configurations by validation loss\n(red = the configuration the coordinate-descent search picked; "
               "it is not always the single lowest-loss config tried, since the search is greedy and doesn't test every combination)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "01_top_configs_mean_std_val_loss.png"), dpi=300)
    plt.close()


def plot_config_val_loss_curves(curves_cache, final_config_key):
    for i, (key, entry) in enumerate(curves_cache.items()):
        config, seed_curves = entry["config"], entry["seed_curves"]; is_best = (key == final_config_key)
        per_seed_best = [min(curve) for _, curve in seed_curves]; mean_best = float(np.mean(per_seed_best))
        plt.figure(figsize=(8, 5))
        for seed, val_loss_curve in seed_curves:
            epochs = np.arange(1, len(val_loss_curve) + 1); plt.plot(epochs, val_loss_curve, label=f"seed {seed}")
        star = " \u2605 BEST (selected by search)" if is_best else ""
        plt.xlabel("Epoch"); plt.ylabel("Validation loss")
        plt.title(f"Config {i}{star}\n{config_label(config)}\nmean best val_loss across seeds = {mean_best:.5f}")
        plt.legend(fontsize=7); plt.tight_layout()
        plt.savefig(os.path.join(CURVES_DIR, f"config_{i:03d}_val_loss_curve.png"), dpi=300); plt.close()
    print(f"Saved {len(curves_cache)} per-configuration validation loss curves in: {CURVES_DIR}")


def plot_random_best_config_predictions(Y_true, Y_pred, best_config, test_dataset_ids,
                                        n_examples=4, random_seed=123):
    n_examples = min(n_examples, len(Y_true))
    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(len(Y_true), size=n_examples, replace=False)
    prediction_rows = []

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = np.asarray(axes).reshape(-1)

    for plot_number, (ax, test_index) in enumerate(zip(axes, selected_indices), start=1):
        y_true_i = Y_true[test_index, :, 0]
        y_pred_i = Y_pred[test_index, :, 0]
        example_mae = mean_absolute_error(y_true_i, y_pred_i)
        example_rmse = np.sqrt(mean_squared_error(y_true_i, y_pred_i))
        dataset_id = test_dataset_ids[test_index]

        ax.plot(y_true_i, label="True temperature", linewidth=2)
        ax.plot(y_pred_i, label="Predicted temperature", linewidth=2)
        ax.set_xlabel("Time step"); ax.set_ylabel("Temperature")
        ax.set_title(f"Random example {plot_number}: test index={test_index}, dataset ID={dataset_id}\n"
                     f"MAE={example_mae:.5f}, RMSE={example_rmse:.5f}")
        ax.legend()

        prediction_rows.append({"plot_number": plot_number, "test_index": int(test_index),
                                "dataset_id": dataset_id, "mae": example_mae, "rmse": example_rmse})

        plt.figure(figsize=(10, 5))
        plt.plot(y_true_i, label="True temperature", linewidth=2)
        plt.plot(y_pred_i, label="Predicted temperature", linewidth=2)
        plt.xlabel("Time step"); plt.ylabel("Temperature")
        plt.title(f"Best configuration - random example {plot_number}\n"
                  f"test index={test_index}, dataset ID={dataset_id}, MAE={example_mae:.5f}\n"
                  f"{config_label(best_config)}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(PREDICTION_DIR, f"random_example_{plot_number:02d}_test_{test_index}.png"), dpi=300)
        plt.close()

    for ax in axes[n_examples:]:
        ax.axis("off")

    fig.suptitle(f"Four random test sequences predicted by the selected best configuration\n{config_label(best_config)}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(PREDICTION_DIR, "four_random_predictions_combined.png"), dpi=300)
    plt.close(fig)

    pd.DataFrame(prediction_rows).to_csv(os.path.join(PREDICTION_DIR, "random_prediction_metrics.csv"), index=False)
    print("Random test indices:", selected_indices.tolist())
    print(f"Saved {n_examples} best-configuration prediction plots in: {PREDICTION_DIR}")


def calculate_common_metrics(Y_true, Y_pred):
    mse = mean_squared_error(Y_true.flatten(), Y_pred.flatten())
    return {
        "mae": mean_absolute_error(Y_true.flatten(), Y_pred.flatten()),
        "rmse": np.sqrt(mse),
    }


def plot_loss_metric_comparison(metrics_df):
    labels = [LOSS_DISPLAY_NAMES[name] for name in metrics_df["loss_name"]]
    x = np.arange(len(labels)); width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split_name, title in zip(
        axes,
        ["validation", "test"],
        ["Validation metrics (used to select loss)", "Test metrics (final evaluation)"],
    ):
        mae_values = metrics_df[f"{split_name}_mae"].to_numpy()
        rmse_values = metrics_df[f"{split_name}_rmse"].to_numpy()
        mae_bars = ax.bar(x - width / 2, mae_values, width, label="MAE")
        rmse_bars = ax.bar(x + width / 2, rmse_values, width, label="RMSE")
        ax.bar_label(mae_bars, fmt="%.4f", padding=3, fontsize=8)
        ax.bar_label(rmse_bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("Temperature error (original units)")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("Loss-function comparison using the same architecture, data split, and random seed\nLower is better")
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(os.path.join(LOSS_COMPARISON_DIR, "loss_comparison_mae_rmse.png"), dpi=300)
    plt.close(fig)


def plot_loss_prediction_comparison(Y_true, predictions_by_loss, test_dataset_ids,
                                    n_examples=4, random_seed=123):
    n_examples = min(n_examples, len(Y_true), 4)
    rng = np.random.default_rng(random_seed)
    selected_indices = rng.choice(len(Y_true), size=n_examples, replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = np.asarray(axes).reshape(-1)

    for plot_number, (ax, test_index) in enumerate(zip(axes, selected_indices), start=1):
        y_true_i = Y_true[test_index, :, 0]
        ax.plot(y_true_i, color="black", linewidth=2.5, label="True temperature")

        for loss_name, Y_pred in predictions_by_loss.items():
            y_pred_i = Y_pred[test_index, :, 0]
            sequence_mae = mean_absolute_error(y_true_i, y_pred_i)
            ax.plot(y_pred_i, linewidth=1.5,
                    label=f"{LOSS_DISPLAY_NAMES[loss_name]} (MAE={sequence_mae:.4f})")

        ax.set_xlabel("Time step"); ax.set_ylabel("Temperature")
        ax.set_title(f"Random example {plot_number}: test index={test_index}, "
                     f"dataset ID={test_dataset_ids[test_index]}")
        ax.legend(fontsize=8)

    for ax in axes[n_examples:]:
        ax.axis("off")

    fig.suptitle("Prediction comparison for Huber, MAE, MSE, and Log-Cosh losses\n"
                 "All models use the same selected architecture and seed", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(LOSS_COMPARISON_DIR, "loss_prediction_comparison_four_examples.png"), dpi=300)
    plt.close(fig)


def compare_loss_functions(data, best_config, huber_model, huber_history,
                           huber_val_objective, huber_test_prediction):
    """
    Stage 2: keep the selected architecture fixed and change only the loss.

    Raw training losses are not compared because Huber, MAE, MSE, and Log-Cosh
    have different numerical scales. Every model is instead compared using MAE
    and RMSE after converting temperatures back to their original units.
    """
    Y_val_true = data["Y_val"] * data["Y_std"] + data["Y_mean"]
    Y_test_true = data["Y_test"] * data["Y_std"] + data["Y_mean"]
    metrics_rows = []
    predictions_by_loss = {}

    for loss_name in LOSS_FUNCTION_OPTIONS:
        print("\n" + "=" * 70)
        print(f"LOSS COMPARISON: {LOSS_DISPLAY_NAMES[loss_name]}")
        print("=" * 70)

        if loss_name == "huber":
            model = huber_model
            history = huber_history
            best_objective_val_loss = huber_val_objective
            Y_test_pred = huber_test_prediction
        else:
            model, history, best_objective_val_loss, Y_test_pred = train_and_predict(
                data=data,
                config=best_config,
                seed=DATA_SEED,
                epochs=FINAL_EPOCHS,
                verbose=1,
                loss_name=loss_name,
            )

        Y_val_pred_norm = model.predict(data["X_val"], verbose=0)
        Y_val_pred = Y_val_pred_norm * data["Y_std"] + data["Y_mean"]
        val_metrics = calculate_common_metrics(Y_val_true, Y_val_pred)
        test_metrics = calculate_common_metrics(Y_test_true, Y_test_pred)

        metrics_rows.append({
            "loss_name": loss_name,
            "display_name": LOSS_DISPLAY_NAMES[loss_name],
            "seed": DATA_SEED,
            "best_own_validation_loss": best_objective_val_loss,
            "validation_mae": val_metrics["mae"],
            "validation_rmse": val_metrics["rmse"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "epochs_trained": len(history.history["loss"]),
        })
        predictions_by_loss[loss_name] = Y_test_pred
        model.save(os.path.join(LOSS_MODELS_DIR, f"model_{loss_name}.keras"))

        print(f"Validation MAE:  {val_metrics['mae']:.6f}")
        print(f"Validation RMSE: {val_metrics['rmse']:.6f}")
        print(f"Test MAE:        {test_metrics['mae']:.6f}")
        print(f"Test RMSE:       {test_metrics['rmse']:.6f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        by=["validation_mae", "validation_rmse"]
    ).reset_index(drop=True)
    metrics_df.to_csv(os.path.join(LOSS_COMPARISON_DIR, "loss_comparison_metrics.csv"), index=False)

    best_loss_name = metrics_df.iloc[0]["loss_name"]
    with open(os.path.join(LOSS_COMPARISON_DIR, "selected_loss.txt"), "w") as f:
        f.write("Loss selected using the lowest validation MAE\n")
        f.write(f"Selected loss: {LOSS_DISPLAY_NAMES[best_loss_name]} ({best_loss_name})\n")
        f.write(f"Configuration: {config_label(best_config)}\n")

    plot_loss_metric_comparison(metrics_df)
    plot_loss_prediction_comparison(
        Y_true=Y_test_true,
        predictions_by_loss=predictions_by_loss,
        test_dataset_ids=data["test_dataset_ids"],
        n_examples=N_RANDOM_PREDICTION_EXAMPLES,
        random_seed=RANDOM_EXAMPLE_SEED,
    )

    print("\nLoss comparison ranked by validation MAE:")
    print(metrics_df[["display_name", "validation_mae", "validation_rmse", "test_mae", "test_rmse"]])
    print(f"\nSelected loss: {LOSS_DISPLAY_NAMES[best_loss_name]}")
    print("Saved loss-comparison results in:", LOSS_COMPARISON_DIR)
    return metrics_df, predictions_by_loss, best_loss_name


def evaluate_and_plot(final_model, results_df, curves_cache, final_config_key, best_config, data, Y_true, Y_pred_best):
    mse = mean_squared_error(Y_true.flatten(), Y_pred_best.flatten()); mae = mean_absolute_error(Y_true.flatten(), Y_pred_best.flatten()); rmse = np.sqrt(mse)
    print("\nTest metrics (best/selected configuration):"); print("Test MSE:", mse); print("Test RMSE:", rmse); print("Test MAE:", mae)
    with open(os.path.join(RESULTS_DIR, "final_test_metrics.txt"), "w") as f:
        f.write(f"Test MSE: {mse}\n"); f.write(f"Test RMSE: {rmse}\n"); f.write(f"Test MAE: {mae}\n")
    np.save(os.path.join(RESULTS_DIR, "Y_true_test.npy"), Y_true); np.save(os.path.join(RESULTS_DIR, "Y_pred_test.npy"), Y_pred_best)
    plot_top_configs(results_df, final_config_key, top_n=10)
    plot_config_val_loss_curves(curves_cache, final_config_key)
    plot_random_best_config_predictions(Y_true=Y_true, Y_pred=Y_pred_best, best_config=best_config,
                                        test_dataset_ids=data["test_dataset_ids"],
                                        n_examples=N_RANDOM_PREDICTION_EXAMPLES,
                                        random_seed=RANDOM_EXAMPLE_SEED)
    print("\nSaved all plots and results in:", RESULTS_DIR)


def main():
    data = prepare_dataset()
    results_df, curves_cache, best_config = run_hyperparameter_search(data)
    final_config_key = config_key(best_config)
    print("\n===================================================="); print("Best configuration selected:"); print(best_config); print("====================================================")
    Y_true = data["Y_test"] * data["Y_std"] + data["Y_mean"]
    print(f"\nTraining selected best configuration with Huber loss:\n{config_label(best_config)}")
    final_model, history, val_loss, Y_pred_best = train_and_predict(
        data, best_config, seed=DATA_SEED, epochs=FINAL_EPOCHS, verbose=1,
        loss_name=SEARCH_LOSS_NAME
    )
    final_model.save(os.path.join(RESULTS_DIR, "best_satis_conv_bilstm_model.keras"))
    print(f"Best Huber validation loss in final training: {val_loss:.6f}")
    evaluate_and_plot(final_model=final_model, results_df=results_df, curves_cache=curves_cache,
                      final_config_key=final_config_key, best_config=best_config, data=data,
                      Y_true=Y_true, Y_pred_best=Y_pred_best)
    compare_loss_functions(
        data=data,
        best_config=best_config,
        huber_model=final_model,
        huber_history=history,
        huber_val_objective=val_loss,
        huber_test_prediction=Y_pred_best,
    )


if __name__ == "__main__":
    main()