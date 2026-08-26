import numpy as np
import pandas as pd


def make_sensor(global_temperature, noise_level=0.2):
    N = len(global_temperature)

    a = np.random.uniform(-1.5, 1.5)
    b = np.random.uniform(0.5, 1.8)

    sensor = a + b * global_temperature
    sensor += noise_level * np.random.randn(N)

    return sensor


def corrupt_sensor(
    sensor,
    n_offset_jumps=2,
    offset_jump_strength=1,
    n_spikes=3,
    n_gaps=2,
    min_gap=8,
    max_gap=25,
):
    corrupted = sensor.copy()
    N = len(sensor)

    jump_points = np.random.choice(
        np.arange(10, N - 10),
        size=n_offset_jumps,
        replace=False,
    )

    jump_points = np.sort(jump_points)

    for jp in jump_points:
        jump_value = np.random.uniform(
            -offset_jump_strength,
            offset_jump_strength,
        )
        corrupted[jp:] += jump_value

    for _ in range(n_spikes):
        center = np.random.randint(0, N)
        spike_amp = np.random.uniform(-2.0, 2.0)
        width = np.random.randint(2, 8)

        left = max(0, center - width)
        right = min(N, center + width)

        pulse = np.hanning(right - left)
        corrupted[left:right] += spike_amp * pulse

    for _ in range(n_gaps):
        start = np.random.randint(0, N - max_gap)
        length = np.random.randint(min_gap, max_gap)

        corrupted[start:start + length] = np.nan

    return corrupted


def generate_one_dataset(seed=None, n_points=300):
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, 50, n_points)
    N = len(t)

    base_periodic = 0.05 * np.sin(2 * np.pi * 0.5 * t)

    random_steps = np.random.normal(0, 0.18, N)
    random_walk = np.cumsum(random_steps)

    shock = np.zeros(N)

    for _ in range(6):
        start = np.random.randint(10, N - 30)
        length = np.random.randint(15, 45)
        end = min(start + length, N)

        shock[start:end] += np.linspace(
            0,
            np.random.uniform(-2.0, 2.0),
            end - start,
        )

    global_temperature = base_periodic + random_walk + shock

    signal2_clean = make_sensor(global_temperature, noise_level=0.15)
    signal3_clean = make_sensor(global_temperature, noise_level=0.20)
    signal4_clean = make_sensor(global_temperature, noise_level=0.25)

    signal2_noisy = corrupt_sensor(signal2_clean)
    signal3_noisy = corrupt_sensor(signal3_clean)
    signal4_noisy = corrupt_sensor(signal4_clean)

    return pd.DataFrame(
        {
            "time": t,
            "global_temperature": global_temperature,

            "signal2_clean": signal2_clean,
            "signal2_noisy": signal2_noisy,

            "signal3_clean": signal3_clean,
            "signal3_noisy": signal3_noisy,

            "signal4_clean": signal4_clean,
            "signal4_noisy": signal4_noisy,
        }
    )


def generate_datasets(
    n_datasets=1000,
    seed=None,
    n_points=200,
):
    datasets = []

    for i in range(n_datasets):
        current_seed = None if seed is None else seed + i

        df_i = generate_one_dataset(
            seed=current_seed,
            n_points=n_points,
        )

        df_i["dataset_id"] = i
        datasets.append(df_i)

    return pd.concat(datasets, ignore_index=True)


def generate_ml_dataset(
    n_datasets=1000,
    seed=None,
    n_points=200,
    input_columns=None,
    target_columns=None,
):
    if input_columns is None:
        input_columns = [
            "signal2_noisy",
            "signal3_noisy",
            "signal4_noisy",
        ]

    if target_columns is None:
        target_columns = [
            "global_temperature",
        ]

    df = generate_datasets(
        n_datasets=n_datasets,
        seed=seed,
        n_points=n_points,
    )

    dataset_ids = sorted(df["dataset_id"].unique())

    X_list = []
    Y_list = []

    for dataset_id in dataset_ids:
        df_i = df[df["dataset_id"] == dataset_id].sort_values("time")

        X_raw = df_i[input_columns].values

        mask = (~np.isnan(X_raw)).astype(float)

        col_means = np.nanmean(X_raw, axis=0)
        X_filled = np.where(np.isnan(X_raw), col_means, X_raw)

        X_model = np.concatenate([X_filled, mask], axis=1)

        Y_raw = df_i[target_columns].values

        X_list.append(X_model)
        Y_list.append(Y_raw)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    return X, Y, dataset_ids, df


def save_datasets(
    filename,
    n_datasets=1000,
    seed=None,
    n_points=200,
):
    df = generate_datasets(
        n_datasets=n_datasets,
        seed=seed,
        n_points=n_points,
    )

    df.to_csv(filename, index=False)

    return df


if __name__ == "__main__":
    df = save_datasets(
        filename="sensor_data.csv",
        n_datasets=1000,
        seed=42,
    )

    print(df.shape)