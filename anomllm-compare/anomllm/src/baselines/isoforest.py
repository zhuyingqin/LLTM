import argparse
from utils import vector_to_interval
from tqdm import trange


def parse_arguments():
    parser = argparse.ArgumentParser(description='Isolation forecast anomaly detection.')
    parser.add_argument('--variant', type=str, default='0shot', help='Variant type')
    parser.add_argument('--model', type=str, default='isolation-forest', help='Model name')
    parser.add_argument('--data', type=str, default='point', help='Data name')
    return parser.parse_args()


def compute_iso_forest_anomalies(series, train_dataset):  # Not using train_dataset
    import numpy as np
    from sklearn.ensemble import IsolationForest

    iso_forest = IsolationForest(random_state=42)
    iso_forest.fit(series)
    anomalies = iso_forest.predict(series)
    iso_forest_anomalies = np.where(anomalies == -1, 1, 0).reshape(-1, 1)
    
    return iso_forest_anomalies


def compute_threshold_anomalies(series, train_dataset):
    import numpy as np
    
    # Calculate the 2nd and 98th percentiles
    lower_threshold = np.percentile(series, 2)
    upper_threshold = np.percentile(series, 98)
    
    # Identify anomalies
    anomalies = np.logical_or(series <= lower_threshold, series >= upper_threshold).astype(float)
    
    return anomalies


def baseline_AD(
    model_name: str,
    data_name: str,
    variant: str,
    num_retries: int = 4,
):
    import json
    import time
    import pickle
    import os
    from loguru import logger
    from data.synthetic import SyntheticDataset

    # Initialize dictionary to store results
    results = {}

    # Configure logger
    log_fn = f"logs/synthetic/{data_name}/{model_name}/" + variant + ".log"
    logger.add(log_fn, format="{time} {level} {message}", level="INFO")
    results_dir = f'results/synthetic/{data_name}/{model_name}/'
    data_dir = f'data/synthetic/{data_name}/eval/'
    train_dir = f'data/synthetic/{data_name}/train/'
    jsonl_fn = os.path.join(results_dir, variant + '.jsonl')
    os.makedirs(results_dir, exist_ok=True)

    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()

    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()

    # Load existing results if jsonl file exists
    if os.path.exists(jsonl_fn):
        with open(jsonl_fn, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                results[entry['custom_id']] = entry["response"]

    # Loop over image files
    for i in trange(1, len(eval_dataset) + 1):
        custom_id = f"{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
        
        # Skip already processed files
        if custom_id in results:
            continue
        
        if model_name == "isolation-forest":
            response = compute_iso_forest_anomalies(
                eval_dataset.series[i - 1],
                train_dataset
            ).flatten()
        elif model_name == "threshold":
            response = compute_threshold_anomalies(
                eval_dataset.series[i - 1],
                train_dataset
            ).flatten()
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
        response = json.dumps([{'start': start, 'end': end} for start, end in vector_to_interval(response)])
        
        # Write the result to jsonl
        with open(jsonl_fn, 'a') as f:
            json.dump({'custom_id': custom_id, 'response': response}, f)
            f.write('\n')


def main():
    args = parse_arguments()
    baseline_AD(
        model_name=args.model,
        data_name=args.data,
        variant=args.variant,
    )


if __name__ == '__main__':
    main()
