import os
import pickle
import numpy as np
from tqdm import trange
from utils import plot_series_and_predictions
import matplotlib.pyplot as plt
import argparse
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset
import importlib
from typing import Optional
from scipy import stats


def synthetic_dataset_with_point_anomalies(
    n_samples: int = 1000,
    number_of_sensors: int = 5,
    frequency: float = 0.03,
    normal_duration_rate: float = 800.0, 
    anomaly_duration_rate: float = 30.0, 
    minimum_anomaly_duration: int = 5, 
    minimum_normal_duration: int = 200, 
    anomaly_std: float = 0.5,
    ratio_of_anomalous_sensors: float = 0.4,
    seed: Optional[int] = None
) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
    """Generate a synthetic dataset with point anomalies in sine waves for multiple sensors.

    Args:
        n_samples: Total number of samples in the dataset.
        number_of_sensors: The number of sensors in the dataset.
        frequency: Base frequency of the sine waves.
        normal_duration_rate: Average duration between anomalies.
        anomaly_duration_rate: Average duration of an anomalous interval.
        anomaly_std: Standard deviation of the normal distribution for anomalies.
        ratio_of_anomalous_sensors: The ratio of sensors which have anomalies in the test set.
        seed: Random seed for reproducibility.

    Returns:
        dataset: the generated dataset of n_samples.
        anomaly_intervals: List of lists of tuples representing (start, end) of anomaly intervals for each sensor.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate sine waves for each sensor
    t = np.arange(n_samples)
    x = np.array([np.sin(2 * np.pi * (frequency + 0.01 * i) * t) for i in range(number_of_sensors)]).T

    # Initialize test labels
    labels = np.zeros((len(x), number_of_sensors))

    # Determine which sensors will have anomalies
    number_of_sensors_with_anomalies = max(1, int(round(number_of_sensors * ratio_of_anomalous_sensors)))
    sensors_with_anomalies = np.random.choice(number_of_sensors, number_of_sensors_with_anomalies, replace=False)

    anomaly_intervals = [[] for _ in range(number_of_sensors)]

    for sensor in sensors_with_anomalies:
        # Use the add_anomalies_to_univariate_series function to get anomaly locations
        _, intervals = add_anomalies_to_univariate_series(
            x[:, sensor],
            normal_duration_rate=normal_duration_rate,
            anomaly_duration_rate=anomaly_duration_rate,
            anomaly_size_range=(-anomaly_std, anomaly_std),
            minimum_anomaly_duration=minimum_anomaly_duration,
            minimum_normal_duration=minimum_normal_duration
        )
        
        # Inject anomalies based on the intervals
        for start, end in intervals:
            anomaly = np.random.normal(0, anomaly_std, end - start)
            x[start:end, sensor] = anomaly
            labels[start:end, sensor] = 1
            anomaly_intervals[sensor].append((start, end))

    return x, anomaly_intervals


def synthetic_dataset_with_frequency_anomalies(
    n_samples: int = 1000,
    number_of_sensors: int = 5,
    frequency: float = 0.03,
    normal_duration_rate: float = 450.0,  # Increased from 300.0
    anomaly_duration_rate: float = 15.0,  # Increased from 10.0
    minimum_anomaly_duration: int = 7,    # Slightly increased from 5
    minimum_normal_duration: int = 20,    # Increased from 10
    frequency_multiplier: float = 3.0,
    ratio_of_anomalous_sensors: float = 0.4,
    seed: Optional[int] = None
) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
    """Generate a synthetic dataset with frequency anomalies in sine waves for multiple sensors.

    Args:
        n_samples: Total number of samples in the dataset.
        number_of_sensors: The number of sensors in the dataset.
        frequency: Base frequency of the sine waves.
        normal_duration_rate: Average duration between anomalies.
        anomaly_duration_rate: Average duration of an anomalous interval.
        minimum_anomaly_duration: Minimum duration of an anomalous interval.
        minimum_normal_duration: Minimum duration of a normal interval.
        frequency_multiplier: Factor by which to multiply or divide the base frequency for anomalies.
        ratio_of_anomalous_sensors: The ratio of sensors which have anomalies in the test set.
        seed: Random seed for reproducibility.

    Returns:
        dataset: the generated dataset of n_samples.
        anomaly_intervals: List of lists of tuples representing (start, end) of anomaly intervals for each sensor.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples)
    x = np.zeros((n_samples, number_of_sensors))
    
    # Initialize test labels
    labels = np.zeros((n_samples, number_of_sensors))

    # Determine which sensors will have anomalies
    number_of_sensors_with_anomalies = max(1, int(round(number_of_sensors * ratio_of_anomalous_sensors)))
    sensors_with_anomalies = np.random.choice(number_of_sensors, number_of_sensors_with_anomalies, replace=False)

    anomaly_intervals = [[] for _ in range(number_of_sensors)]

    for sensor in range(number_of_sensors):
        base_freq = frequency + 0.01 * sensor
        freq_function = np.full(n_samples, base_freq)

        if sensor in sensors_with_anomalies:
            current_time = 0
            while current_time < n_samples:
                normal_duration = max(minimum_normal_duration, int(np.random.exponential(normal_duration_rate)))
                current_time += normal_duration

                if current_time >= n_samples:
                    break

                anomaly_duration = max(minimum_anomaly_duration, int(np.random.exponential(anomaly_duration_rate)))
                anomaly_end = min(n_samples, current_time + anomaly_duration)
                
                # Randomly choose to increase or decrease frequency
                if np.random.random() < 0.5:
                    freq_function[current_time:anomaly_end] *= frequency_multiplier
                else:
                    freq_function[current_time:anomaly_end] /= frequency_multiplier

                labels[current_time:anomaly_end, sensor] = 1
                anomaly_intervals[sensor].append((current_time, anomaly_end))
                current_time = anomaly_end

        # Generate the sine wave with varying frequency
        dx = np.full_like(t, 1.0)
        x_plot = (freq_function * dx).cumsum()
        x[:, sensor] = np.sin(2 * np.pi * x_plot)

    return x, anomaly_intervals


def synthetic_dataset_with_trend_anomalies(
    n_samples: int = 1000,
    number_of_sensors: int = 5,
    frequency: float = 0.02,
    normal_duration_rate: float = 1700.0,
    anomaly_duration_rate: float = 100.0,
    minimum_anomaly_duration: int = 50,
    minimum_normal_duration: int = 800,
    ratio_of_anomalous_sensors: float = 0.4,
    normal_slope: float = 3.0,
    abnormal_slope_range: tuple[float, float] = (6.0, 20.0),
    inverse_ratio: float = 0.0,
    seed: Optional[int] = None
) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
    """Generate a synthetic dataset with trend anomalies in sine waves for multiple sensors.

    Args:
        n_samples: Total number of samples in the dataset.
        number_of_sensors: The number of sensors in the dataset.
        frequency: Base frequency of the sine waves.
        normal_duration_rate: Average duration between anomalies.
        anomaly_duration_rate: Average duration of an anomalous interval.
        minimum_anomaly_duration: Minimum duration of an anomalous interval.
        minimum_normal_duration: Minimum duration of a normal interval.
        ratio_of_anomalous_sensors: The ratio of sensors which have anomalies in the test set.
        normal_slope: The slope of the normal trend.
        abnormal_slope_range: The range of slopes for abnormal trends (min, max).
        inverse_ratio: The ratio of slopes that have different signs (positive/negative).
        seed: Random seed for reproducibility.

    Returns:
        dataset: the generated dataset of n_samples
        anomaly_intervals: List of lists of tuples representing (start, end) of anomaly intervals for each sensor.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(n_samples)
    x = np.zeros((n_samples, number_of_sensors))
    
    # Determine which sensors will have anomalies
    number_of_sensors_with_anomalies = max(1, int(round(number_of_sensors * ratio_of_anomalous_sensors)))
    sensors_with_anomalies = np.random.choice(number_of_sensors, number_of_sensors_with_anomalies, replace=False)

    anomaly_intervals = [[] for _ in range(number_of_sensors)]

    for sensor in range(number_of_sensors):
        base_freq = frequency + 0.01 * sensor
        trend = np.zeros(n_samples)
        current_value = 0.0
        current_time = 0

        if sensor in sensors_with_anomalies:
            # Generate anomaly intervals for the test set
            _, intervals = add_anomalies_to_univariate_series(
                np.zeros(n_samples),  # Dummy series, we only need the intervals
                normal_duration_rate=normal_duration_rate,
                anomaly_duration_rate=anomaly_duration_rate,
                anomaly_size_range=(0, 1),  # Dummy range, not used
                minimum_anomaly_duration=minimum_anomaly_duration,
                minimum_normal_duration=minimum_normal_duration
            )
            
            for start, end in intervals:
                # Normal trend before anomaly
                trend[current_time:start] = current_value + normal_slope * (t[current_time:start] - t[current_time]) / n_samples
                current_value = trend[start - 1]
                # Abnormal trend during anomaly
                abnormal_slope = generate_abnormal_slope(normal_slope, abnormal_slope_range, inverse_ratio)
                trend[start:end] = current_value + abnormal_slope * (t[start:end] - t[start]) / n_samples
                current_value = trend[end - 1]
                current_time = end
                anomaly_intervals[sensor].append((start, end))

        # Normal trend after last anomaly
        if current_time < n_samples:
            trend[current_time:] = current_value + normal_slope * (t[current_time:] - t[current_time]) / n_samples

        # Generate the sine wave with the trend
        x[:, sensor] = np.sin(2 * np.pi * base_freq * t) + trend
        
        # Normalize the series to be between -1 and 1
        x[:, sensor] = 2 * (x[:, sensor] - np.min(x[:, sensor])) / (np.max(x[:, sensor]) - np.min(x[:, sensor])) - 1

    return x, anomaly_intervals


def synthetic_dataset_with_flat_trend_anomalies(**args):
    return synthetic_dataset_with_trend_anomalies(
        normal_slope=3.0,
        abnormal_slope_range=(4.5, 6.0),
        inverse_ratio=0.0,
        **args
    )


def generate_abnormal_slope(normal_slope: float, abnormal_slope_range: tuple[float, float], inverse_ratio: float) -> float:
    """Generate an abnormal slope based on the normal slope and the specified range."""
    min_slope, max_slope = abnormal_slope_range
    if np.isinf(max_slope):
        max_slope = max(abs(normal_slope) * 10, min_slope * 2)  # Set a reasonable upper bound

    if np.random.random() > inverse_ratio:  # 50% chance for a slope above the range
        return np.random.uniform(max(normal_slope, min_slope), max_slope)
    else:  # 50% chance for a slope below the range (including negative)
        lower_bound = min(-max_slope, min(normal_slope, min_slope))
        upper_bound = 0.0
        return np.random.uniform(lower_bound, upper_bound)


def add_anomalies_to_univariate_series(
    x: np.ndarray,
    normal_duration_rate: float,
    anomaly_duration_rate: float,
    anomaly_size_range: tuple[float, float],
    minimum_anomaly_duration: int,
    minimum_normal_duration: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Add anomalies to a given time series.

    Args:
        x: The series to add anomalies to.
        normal_duration_rate: Average duration of a normal interval.
        anomaly_duration_rate: Average duration of an anomalous interval.
        anomaly_size_range: A range where the magnitude of the anomaly lies.
            E.g. if this is (0.5, 0.8), then a random value in that interval with be
            added or subtracted from the series in the anomaly interval.

    Returns:
        x: A copy of the original array which has anomalies added to it.
        anomaly_intervals: A list of tuples which represent the (start, end) of the anomaly intervals.
    """
    # Validate the anomaly size range.
    if anomaly_size_range[0] >= anomaly_size_range[1]:
        raise ValueError(
            f"The anomaly size range {anomaly_size_range} should be strictly increasing."
        )

    # Copy x in order to not overwrite it.
    x = x.copy()
    N = len(x)
    # Define two exponential distributions which describe the lengths of normal and anomalous intervals.
    # So e.g. stats.expon(scale=20) will sample a duration of an anomalous interval with mean 20.
    distr_duration_normal = stats.expon(scale=normal_duration_rate)
    distr_duration_anomalous = stats.expon(scale=anomaly_duration_rate)

    # Loop over a max number of intervals and add the anomalies.
    max_number_of_intervals = 8
    location = 0
    anomaly_intervals = []
    for _ in range(max_number_of_intervals):
        # First sample a normal interval. The anomaly will start at the end of it.
        random_states = np.random.randint(0, np.iinfo(np.int32).max, size=2)
        norm_dur = distr_duration_normal.rvs(random_state=random_states[0])
        norm_dur = max(norm_dur, minimum_normal_duration)
        
        anom_start = location + int(norm_dur)
        anom_dur = distr_duration_anomalous.rvs(random_state=random_states[1])
        anom_dur = max(anom_dur, minimum_anomaly_duration)
        
        # Then sample an anomalous interval. The anomaly will end at the end of it.
        anom_end = anom_start + int(anom_dur)
        
        # Make sure we don't exceed the length of the series.
        anom_end = min(N, anom_end)

        if anom_start >= N:
            break

        # The anomaly shifts the signal up or down to the interval [-0.8, -0.5] or [0.5, 0.8].
        shift_sign = 1 if np.random.randint(low=0, high=2) == 1 else -1
        shift = shift_sign * np.random.uniform(
            anomaly_size_range[0], anomaly_size_range[1], size=anom_end - anom_start
        )
        x[anom_start:anom_end] += shift
        # Update the location to the end of the anomaly.
        location = anom_end

        # mark the indices of anomaly for creating labels
        anomaly_intervals.append((anom_start, anom_end))

    return x, anomaly_intervals


# This function is adapted from the QuoVadis TAD project
# Author: S. Sarfraz
# Source: https://github.com/ssarfraz/QuoVadisTAD.git
# License: MIT
def synthetic_dataset_with_out_of_range_anomalies(
    number_of_sensors: int = 1,
    train_size: int = 5_000,
    test_size: int = 1000,
    nominal_data_mean: float = 0.0,
    nominal_data_std: float = 0.1,
    normal_duration_rate: float = 800.0,
    anomaly_duration_rate: float = 20.0,
    anomaly_size_range: tuple = (0.5, 0.8),
    minimum_anomaly_duration: int = 5,
    minimum_normal_duration: int = 10,
    ratio_of_anomalous_sensors: float = 0.2,
    seed: Optional[int] = None
) -> tuple[dict[str, np.ndarray], list[list[tuple[int, int]]]]:
    """Generate a synthetic dataset with out-of-range anomalies. Normal data are i.i.d. distributed in time based on
    a normal distribution. The test data are generated the same way and then anomalies are added to some randomly
    selected sensors. The anomalies appear as shifts away of the mean of the normal distribution in some intervals
    whose starts are selected based on an exponential distribution. All those parameters can be controlled in the
    function input and are set to some reasonable defaults.

    Args:
        number_of_sensors: The number of sensors of the dataset. To generate univariate datasets, just set this to 1.
        train_size: The size of the nominal training series in timestamps.
        test_size: The size of the anomalous test series in timestamps.
        nominal_data_mean: The mean of the normal distribution defining nominal data.
        nominal_data_std: The standard deviation of the normal distribution defining nominal data.
        normal_duration_rate: Average duration of a normal interval in the anomalous test data.
        anomaly_duration_rate: Average duration of an anomalous interval in the anomalous test data.
        anomaly_size_range: A range where the magnitude of the anomaly lies.
            E.g. if this is (0.5, 0.8), then a random value in that interval with be
            added or subtracted from the series in the anomaly interval.
        ratio_of_anomalous_sensors: The ratio of sensors which have anomalies in the test set.
        seed: Random seed for reproducibility.

    Returns:
        dataset: A dictionary of the form {'train': train, 'test': test, 'labels': labels} containing all the
            information of the generated dataset.
        anomaly_intervals: Lists of tuples which represent the (start, end) of the anomaly intervals. They are in a
            dictionary which maps the anomalous sensor indices to the corresponding anomaly intervals.
    """
    # Fix the random state of numpy.
    np.random.seed(seed)

    # Generate the nominal train data. Just a multivariate series of length `train_size` with `number_of_sensors`
    # features which are independently sampled from the same normal distribution.
    train = np.random.normal(
        nominal_data_mean,
        nominal_data_std,
        size=(train_size, number_of_sensors)
    )

    # Generate the test data the same way as the train data.
    test = np.random.normal(
        nominal_data_mean,
        nominal_data_std,
        size=(test_size, number_of_sensors)
    )

    # Add some anomalies to randomly selected sensors.
    number_of_sensors_with_anomalies = max(1, int(round(number_of_sensors * ratio_of_anomalous_sensors)))
    sensors_with_anomalies = np.random.choice(number_of_sensors, number_of_sensors_with_anomalies, replace=False)

    # Create labels which capture the anomalies. Also capture the locations as intervals for visualization purposes.
    all_locations = {}
    labels = np.zeros_like(test)
    for idx in sensors_with_anomalies:
        test[:, idx], anomaly_locations = add_anomalies_to_univariate_series(
            test[:, idx],
            normal_duration_rate=normal_duration_rate,
            anomaly_duration_rate=anomaly_duration_rate,
            anomaly_size_range=anomaly_size_range,
            minimum_anomaly_duration=minimum_anomaly_duration,
            minimum_normal_duration=minimum_normal_duration
        )

        for start, end in anomaly_locations:
            labels[start:end, idx] = 1

        all_locations[idx] = anomaly_locations

    dataset = {'train': train, 'test': test, 'labels': labels}
    anomaly_intervals = [all_locations.get(i, []) for i in range(test.shape[1])]

    return dataset['test'], anomaly_intervals


class SyntheticDataset(Dataset):

    def __init__(
        self,
        data_dir="data/synthetic/range/",
        synthetic_func_name="synthetic_dataset_with_out_of_range_anomalies",
    ):
        self.data_dir = data_dir
        self.figs_dir = os.path.join(data_dir, 'figs')
        self.series = []
        self.anom = []
        
        # Load the function dynamically
        self.synthetic_func = globals()[synthetic_func_name]

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)

    def generate(self, num_series=400, seed=42, add_noise=False):
        # Fix the seed for reproducibility
        np.random.seed(seed)

        # Generate series
        for i in trange(num_series):
            data, anomaly_locations = self.synthetic_func(
                number_of_sensors=1,
                ratio_of_anomalous_sensors=1.0
            )
            if add_noise:
                data += np.random.normal(0, 0.08, data.shape)
            self.series.append(data)
            self.anom.append(anomaly_locations)
            
            # Plot and save the figure
            fig = plot_series_and_predictions(
                series=data, 
                single_series_figsize=(10, 1.5),
                gt_anomaly_intervals=anomaly_locations,
                anomalies=None
            )
            fig_path = os.path.join(self.figs_dir, f'{i + 1:03d}.png')
            fig.savefig(fig_path)
            plt.close()

        # Save the data
        self.save()

    def save(self):
        data_dict = {
            'series': self.series,
            'anom': self.anom
        }
        with open(os.path.join(self.data_dir, 'data.pkl'), 'wb') as f:
            pickle.dump(data_dict, f)

    def load(self):
        # Load data
        with open(os.path.join(self.data_dir, 'data.pkl'), 'rb') as f:
            data_dict = pickle.load(f)
        self.series = data_dict['series']
        self.anom = data_dict['anom']
        self.name = os.path.basename(os.path.dirname(os.path.dirname(self.data_dir)))
        print(f"Loaded dataset {self.name} with {len(self.series)} series.")

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        anom = self.anom[idx]
        series = self.series[idx]

        # Convert to torch tensors
        anom = torch.tensor(anom, dtype=torch.float32)
        series = torch.tensor(series, dtype=torch.float32)

        return anom, series
    
    def few_shots(self, num_shots=5, idx=None):
        if idx is None:
            idx = np.random.choice(len(self.series), num_shots, replace=False)
        few_shot_data = []
        for i in idx:
            anom, series = self.__getitem__(i)
            anom = [{"start": int(start.item()), "end": int(end.item())} for start, end in list(anom[0])]
            few_shot_data.append((series, anom))
        return few_shot_data


def main(args):
    dataset = SyntheticDataset(args.data_dir, args.synthetic_func)
    if args.generate:
        dataset.generate(args.num_series, args.seed, args.add_noise)
    else:
        dataset.load()
    
    print(f"Dataset loaded with {len(dataset.series)} series.")
    print(f"Number of anomaly lists: {len(dataset.anom)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate or load synthetic dataset with out-of-range anomalies")
    parser.add_argument("--num_series", type=int, default=400, help="Number of series to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, default='data/synthetic/range/', help="Directory to save/load the data")
    parser.add_argument("--generate", action="store_true", help="Generate new data instead of loading existing data")
    parser.add_argument("--add_noise", action="store_true", help="Add noise to the generated data")
    parser.add_argument("--synthetic_func", type=str, default="synthetic_dataset_with_out_of_range_anomalies", 
                        help="Name of the synthetic function to use")
    
    args = parser.parse_args()
    main(args)
