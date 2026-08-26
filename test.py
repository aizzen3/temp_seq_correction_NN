import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, LSTM, GRU, Multiply, Concatenate, Softmax, Conv1D, Bidirectional,
                                      Dropout, LayerNormalization, Lambda, Add, MultiHeadAttention, Embedding,
                                      ZeroPadding1D, Cropping1D, Reshape)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sensor_simulator import generate_ml_dataset

RESULTS_DIR = "satis_conv_search_configuration_test"; os.makedirs(RESULTS_DIR, exist_ok=True)
CURVES_DIR = os.path.join(RESULTS_DIR, "config_val_loss_curves"); os.makedirs(CURVES_DIR, exist_ok=True)
PREDICTION_DIR = os.path.join(RESULTS_DIR, "best_config_random_predictions"); os.makedirs(PREDICTION_DIR, exist_ok=True)
PATCH_PREDICTION_DIR = os.path.join(RESULTS_DIR, "patch_transformer_random_predictions"); os.makedirs(PATCH_PREDICTION_DIR, exist_ok=True)
N_DATASETS = 1000; DATA_SEED = 42; SPIKE_THRESHOLD = 5.0
BATCH_SIZE = 32; SEARCH_EPOCHS = 80; FINAL_EPOCHS = 200
N_RANDOM_PREDICTION_EXAMPLES = 4; RANDOM_EXAMPLE_SEED = 123
CONV_FILTER_OPTIONS = [(16,), (32,), (64,), (128,), (256,), (16, 32), (32, 64), (64, 128), (128, 256), (16, 32, 64), (32, 64, 128), (64, 128, 256)]
KERNEL_OPTIONS = [3, 5, 7]; DILATION_OPTIONS = [1, 2, 4]; LSTM_UNITS_OPTIONS = [64, 128]; DROPOUT_OPTIONS = [0.1]; LEARNING_RATE_OPTIONS = [1e-3, 3e-3]

# NEW: learned imputation is on by default for every model built in this file (both backbones).
# imputer_units controls the size of the small BiGRU that predicts a context-aware
# replacement value for whatever the mask marks as missing (spikes / NaNs).
INITIAL_CONFIG = {"conv_filters": (64, 128), "kernel_size": 5, "dilation_rate": 1, "lstm_units": 64, "dropout_rate": 0.1,
                   "learning_rate": 1e-3, "backbone": "conv_bilstm", "use_learned_imputation": True, "imputer_units": 32}
SEARCH_SPACE = {"conv_filters": CONV_FILTER_OPTIONS, "kernel_size": KERNEL_OPTIONS, "dilation_rate": DILATION_OPTIONS, "lstm_units": LSTM_UNITS_OPTIONS, "dropout_rate": DROPOUT_OPTIONS, "learning_rate": LEARNING_RATE_OPTIONS}
MAX_COORDINATE_PASSES = 3; MIN_SCORE_IMPROVEMENT = 1e-5
SEARCH_SEEDS = [1, 2, 3]; STD_WEIGHT = 1.0

# NEW: config used for the standalone patch-transformer comparison run (not part of the
# coordinate-descent search -- see run_patch_transformer_comparison() in main()).
PATCH_TRANSFORMER_CONFIG = {"backbone": "patch_transformer", "patch_len": 8, "d_model": 64, "num_heads": 4,
                             "num_transformer_blocks": 2, "ff_dim": 128, "dropout_rate": 0.1, "learning_rate": 1e-3,
                             "use_learned_imputation": True, "imputer_units": 32}

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
        # NOTE: this mean-fill is now only a *fallback initial value*. Inside the model,
        # build_satis_model() replaces every masked (missing) position with a context-aware
        # value predicted by a small BiGRU, so this mean-fill barely matters anymore -- it's
        # just what the imputer sees as a neutral starting point at missing timesteps.
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


def apply_learned_imputation(sensor_values, sensor_masks, sensor_derivatives, imputer_units):
    """
    Learned, context-aware imputation (BRITS/SAITS-style idea, simplified):
    a small bidirectional GRU reads [values, mask, derivative] across the WHOLE sequence
    (both directions) and, for every timestep, predicts a "candidate" value per sensor.
    That candidate is used ONLY where the mask says the point is missing -- observed
    points are passed through untouched. Because the candidate feeds into the rest of
    the network and is trained end-to-end on the main temperature loss, the imputer
    learns to fill gaps in whatever way actually helps the downstream prediction,
    instead of a fixed column mean.
    """
    impute_context = Concatenate()([sensor_values, sensor_masks, sensor_derivatives])
    impute_hidden = Bidirectional(GRU(imputer_units, return_sequences=True))(impute_context)
    candidate_values = Dense(3, name="imputation_candidate")(impute_hidden)
    observed_part = Multiply()([sensor_values, sensor_masks])
    missing_gate = Lambda(lambda m: 1.0 - m)(sensor_masks)
    candidate_part = Multiply()([candidate_values, missing_gate])
    return Add(name="learned_imputed_sensors")([observed_part, candidate_part])


def build_satis_model(input_shape, conv_filters=(64, 128), kernel_size=5, dilation_rate=1, lstm_units=64, dropout_rate=0.1,
                       learning_rate=1e-3, use_learned_imputation=True, imputer_units=32):
    inputs = Input(shape=input_shape)
    sensor_values = inputs[:, :, :3]; sensor_masks = inputs[:, :, 3:6]; sensor_derivatives = inputs[:, :, 6:9]

    if use_learned_imputation:
        sensor_values = apply_learned_imputation(sensor_values, sensor_masks, sensor_derivatives, imputer_units)

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
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.Huber(delta=1.0), metrics=["mae"])
    return model


def transformer_encoder_block(x, num_heads, key_dim, ff_dim, dropout_rate):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    x1 = LayerNormalization(epsilon=1e-6)(Add()([x, attn_output]))
    ff = Dense(ff_dim, activation="relu")(x1)
    ff = Dense(x1.shape[-1])(ff)
    ff = Dropout(dropout_rate)(ff)
    x2 = LayerNormalization(epsilon=1e-6)(Add()([x1, ff]))
    return x2


def build_satis_patch_transformer_model(input_shape, patch_len=8, d_model=64, num_heads=4, num_transformer_blocks=2,
                                         ff_dim=128, dropout_rate=0.1, learning_rate=1e-3,
                                         use_learned_imputation=True, imputer_units=32):
    """
    PatchTST-style backbone, dropped in as an alternative to the Conv1D+BiLSTM stack.
    Same sensor-attention front end as build_satis_model(), then:
      1) split the (T, 9) feature sequence into non-overlapping patches of length patch_len,
      2) linearly embed each patch to d_model (via a stride=patch_len Conv1D -- this IS the
         standard "patchify + linear projection" step, just expressed as a strided conv),
      3) add a learned positional embedding per patch,
      4) run num_transformer_blocks self-attention encoder blocks over the patch sequence,
      5) project each patch's final representation back down to patch_len scalar outputs
         (one per original timestep) and reshape/un-patch back to the full sequence length.
    T is padded at the front to a multiple of patch_len if it doesn't divide evenly, then
    the padding is cropped back off at the end so the output length always matches Y.
    """
    T, C_in = input_shape
    inputs = Input(shape=input_shape)
    sensor_values = inputs[:, :, :3]; sensor_masks = inputs[:, :, 3:6]; sensor_derivatives = inputs[:, :, 6:9]

    if use_learned_imputation:
        sensor_values = apply_learned_imputation(sensor_values, sensor_masks, sensor_derivatives, imputer_units)

    attention_logits = Dense(3)(inputs)
    mask_penalty = Lambda(lambda m: (1.0 - m) * (-1e9))(sensor_masks)
    attention_logits = Add()([attention_logits, mask_penalty])
    attention_weights = Softmax(axis=-1, name="sensor_attention")(attention_logits)
    weighted_sensors = Multiply()([sensor_values, attention_weights])
    satis_features = Concatenate()([weighted_sensors, sensor_masks, sensor_derivatives])  # (T, 9)

    pad_len = (-T) % patch_len
    x = ZeroPadding1D(padding=(pad_len, 0))(satis_features) if pad_len > 0 else satis_features
    T_padded = T + pad_len
    num_patches = T_padded // patch_len

    x = Conv1D(filters=d_model, kernel_size=patch_len, strides=patch_len, padding="valid", name="patch_embed")(x)  # (num_patches, d_model)

    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = Embedding(input_dim=num_patches, output_dim=d_model)(positions)  # (num_patches, d_model)
    x = Lambda(lambda t: t + position_embedding, name="add_positional_embedding")(x)

    for _ in range(num_transformer_blocks):
        x = transformer_encoder_block(x, num_heads=num_heads, key_dim=max(1, d_model // num_heads), ff_dim=ff_dim, dropout_rate=dropout_rate)

    x = Dense(patch_len, name="patch_to_timesteps")(x)  # (num_patches, patch_len)
    x = Reshape((num_patches * patch_len, 1))(x)  # (T_padded, 1)
    if pad_len > 0:
        x = Cropping1D(cropping=(pad_len, 0))(x)  # (T, 1)
    outputs = x

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=tf.keras.losses.Huber(delta=1.0), metrics=["mae"])
    return model


def build_model_by_backbone(input_shape, config):
    """Dispatches to the right builder based on config['backbone'] ('conv_bilstm' default, or 'patch_transformer')."""
    backbone = config.get("backbone", "conv_bilstm")
    use_learned_imputation = config.get("use_learned_imputation", True)
    imputer_units = config.get("imputer_units", 32)
    if backbone == "patch_transformer":
        return build_satis_patch_transformer_model(
            input_shape=input_shape, patch_len=config.get("patch_len", 8), d_model=config.get("d_model", 64),
            num_heads=config.get("num_heads", 4), num_transformer_blocks=config.get("num_transformer_blocks", 2),
            ff_dim=config.get("ff_dim", 128), dropout_rate=float(config.get("dropout_rate", 0.1)),
            learning_rate=float(config.get("learning_rate", 1e-3)),
            use_learned_imputation=use_learned_imputation, imputer_units=imputer_units)
    return build_satis_model(
        input_shape=input_shape, conv_filters=tuple(config["conv_filters"]), kernel_size=int(config["kernel_size"]),
        dilation_rate=int(config["dilation_rate"]), lstm_units=int(config["lstm_units"]),
        dropout_rate=float(config["dropout_rate"]), learning_rate=float(config["learning_rate"]),
        use_learned_imputation=use_learned_imputation, imputer_units=imputer_units)


def train_one_model(data, config, seed, epochs=80, batch_size=32, verbose=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model_by_backbone(input_shape, config)
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
    if config.get("backbone", "conv_bilstm") == "patch_transformer":
        return (f"backbone=patch_transformer, patch_len={config.get('patch_len')}, d_model={config.get('d_model')}, "
                f"heads={config.get('num_heads')}, blocks={config.get('num_transformer_blocks')}, "
                f"lr={config.get('learning_rate')}, learned_imputation={config.get('use_learned_imputation')}")
    return (f"filters={tuple(config['conv_filters'])}, k={config['kernel_size']}, d={config['dilation_rate']}, "
            f"lstm={config['lstm_units']}, drop={config['dropout_rate']}, lr={config['learning_rate']}, "
            f"learned_imputation={config.get('use_learned_imputation')}")


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


def train_and_predict(data, config, seed=DATA_SEED, epochs=FINAL_EPOCHS, verbose=0):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    X_train, Y_train, X_val, Y_val = data["X_train"], data["Y_train"], data["X_val"], data["Y_val"]
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model_by_backbone(input_shape, config)
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


def plot_random_config_predictions(Y_true, Y_pred, config, test_dataset_ids, prediction_dir,
                                    label_prefix="Best configuration", n_examples=4, random_seed=123):
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
        plt.title(f"{label_prefix} - random example {plot_number}\n"
                  f"test index={test_index}, dataset ID={dataset_id}, MAE={example_mae:.5f}\n"
                  f"{config_label(config)}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(prediction_dir, f"random_example_{plot_number:02d}_test_{test_index}.png"), dpi=300)
        plt.close()

    for ax in axes[n_examples:]:
        ax.axis("off")

    fig.suptitle(f"Four random test sequences predicted by: {label_prefix}\n{config_label(config)}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(prediction_dir, "four_random_predictions_combined.png"), dpi=300)
    plt.close(fig)

    pd.DataFrame(prediction_rows).to_csv(os.path.join(prediction_dir, "random_prediction_metrics.csv"), index=False)
    print("Random test indices:", selected_indices.tolist())
    print(f"Saved {n_examples} prediction plots in: {prediction_dir}")


def evaluate_and_plot(final_model, results_df, curves_cache, final_config_key, best_config, data, Y_true, Y_pred_best):
    mse = mean_squared_error(Y_true.flatten(), Y_pred_best.flatten()); mae = mean_absolute_error(Y_true.flatten(), Y_pred_best.flatten()); rmse = np.sqrt(mse)
    print("\nTest metrics (best/selected configuration):"); print("Test MSE:", mse); print("Test RMSE:", rmse); print("Test MAE:", mae)
    with open(os.path.join(RESULTS_DIR, "final_test_metrics.txt"), "w") as f:
        f.write(f"Test MSE: {mse}\n"); f.write(f"Test RMSE: {rmse}\n"); f.write(f"Test MAE: {mae}\n")
    np.save(os.path.join(RESULTS_DIR, "Y_true_test.npy"), Y_true); np.save(os.path.join(RESULTS_DIR, "Y_pred_test.npy"), Y_pred_best)
    plot_top_configs(results_df, final_config_key, top_n=10)
    plot_config_val_loss_curves(curves_cache, final_config_key)
    plot_random_config_predictions(Y_true=Y_true, Y_pred=Y_pred_best, config=best_config, test_dataset_ids=data["test_dataset_ids"],
                                    prediction_dir=PREDICTION_DIR, label_prefix="Best configuration (conv+BiLSTM search)",
                                    n_examples=N_RANDOM_PREDICTION_EXAMPLES, random_seed=RANDOM_EXAMPLE_SEED)
    return mse, mae, rmse


def run_patch_transformer_comparison(data, Y_true):
    """
    NEW: standalone comparison run. Trains the PatchTST-style backbone (with the same
    learned imputation and sensor-attention front end) on the same data/split, and reports
    its test metrics side by side with the conv+BiLSTM search winner. Not wired into the
    coordinate-descent search itself -- run it after the main search to see whether the
    patch-transformer backbone is worth adding to the search space for your dataset.
    """
    print("\n" + "#" * 70)
    print("Training PATCH-TRANSFORMER backbone for comparison")
    print(f"Config: {config_label(PATCH_TRANSFORMER_CONFIG)}")
    print("#" * 70)
    patch_model, patch_history, patch_val_loss, Y_pred_patch = train_and_predict(
        data, PATCH_TRANSFORMER_CONFIG, seed=DATA_SEED, epochs=FINAL_EPOCHS, verbose=1
    )
    patch_model.save(os.path.join(RESULTS_DIR, "patch_transformer_model.keras"))
    mse_p = mean_squared_error(Y_true.flatten(), Y_pred_patch.flatten())
    mae_p = mean_absolute_error(Y_true.flatten(), Y_pred_patch.flatten())
    rmse_p = np.sqrt(mse_p)
    print(f"\nPatch-transformer test metrics: MSE={mse_p:.6f}, RMSE={rmse_p:.6f}, MAE={mae_p:.6f}")
    with open(os.path.join(RESULTS_DIR, "patch_transformer_test_metrics.txt"), "w") as f:
        f.write(f"Test MSE: {mse_p}\n"); f.write(f"Test RMSE: {rmse_p}\n"); f.write(f"Test MAE: {mae_p}\n")
    plot_random_config_predictions(Y_true=Y_true, Y_pred=Y_pred_patch, config=PATCH_TRANSFORMER_CONFIG, test_dataset_ids=data["test_dataset_ids"],
                                    prediction_dir=PATCH_PREDICTION_DIR, label_prefix="Patch-transformer backbone",
                                    n_examples=N_RANDOM_PREDICTION_EXAMPLES, random_seed=RANDOM_EXAMPLE_SEED)
    return mse_p, mae_p, rmse_p


def main():
    data = prepare_dataset()
    results_df, curves_cache, best_config = run_hyperparameter_search(data)
    final_config_key = config_key(best_config)
    print("\n===================================================="); print("Best configuration selected:"); print(best_config); print("====================================================")
    Y_true = data["Y_test"] * data["Y_std"] + data["Y_mean"]
    print(f"\nTraining selected best configuration for final prediction:\n{config_label(best_config)}")
    final_model, history, val_loss, Y_pred_best = train_and_predict(
        data, best_config, seed=DATA_SEED, epochs=FINAL_EPOCHS, verbose=1
    )
    final_model.save(os.path.join(RESULTS_DIR, "best_satis_conv_bilstm_model.keras"))
    print(f"Best validation loss in final training: {val_loss:.6f}")
    mse, mae, rmse = evaluate_and_plot(final_model=final_model, results_df=results_df, curves_cache=curves_cache,
                                        final_config_key=final_config_key, best_config=best_config, data=data,
                                        Y_true=Y_true, Y_pred_best=Y_pred_best)

    mse_p, mae_p, rmse_p = run_patch_transformer_comparison(data, Y_true)

    print("\n" + "=" * 70)
    print("BACKBONE COMPARISON (test set)")
    print(f"  conv+BiLSTM (search winner) : MSE={mse:.6f}  RMSE={rmse:.6f}  MAE={mae:.6f}")
    print(f"  patch_transformer            : MSE={mse_p:.6f}  RMSE={rmse_p:.6f}  MAE={mae_p:.6f}")
    print("=" * 70)
    print("\nSaved all plots and results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()