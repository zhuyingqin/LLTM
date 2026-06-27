from utils import parse_output
from openai_api import send_openai_request
from config import create_batch_api_configs
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process online API anomaly detection.')
    parser.add_argument('--variant', type=str, default='1shot-vision', help='Variant type')
    parser.add_argument('--model', type=str, default='internvlm-76b', help='Model name')
    parser.add_argument('--data', type=str, default='point', help='Data name')
    return parser.parse_args()


def online_AD_with_retries(
    model_name: str,
    data_name: str,
    request_func: callable,
    variant: str = "standard",
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
    for i in range(1, len(eval_dataset) + 1):
        custom_id = f"{data_name}_{model_name}_{variant}_{str(i).zfill(5)}"
        
        # Skip already processed files
        if custom_id in results:
            continue
        
        # Perform anomaly detection with exponential backoff
        for attempt in range(num_retries):
            try:
                request = request_func(
                    eval_dataset.series[i - 1],
                    train_dataset
                )
                response = send_openai_request(request, model_name)
                # Write the result to jsonl
                with open(jsonl_fn, 'a') as f:
                    json.dump({'custom_id': custom_id, 'request': request, 'response': response}, f)
                    f.write('\n')
                # If successful, break the retry loop
                break
            except Exception as e:
                if "503" in str(e):  # Server not up yet, sleep until the server is up again
                    while True:
                        logger.debug("503 error, sleep 30 seconds")
                        time.sleep(30)
                        try:
                            response = send_openai_request(request, model_name)
                            break
                        except Exception as e:
                            if "503" not in str(e):
                                break
                else:
                    logger.error(e)
                    # If an exception occurs, wait and then retry
                    wait_time = 2 ** (attempt + 3)
                    logger.debug(f"Attempt {attempt + 1} failed. Waiting for {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
        else:
            logger.error(f"Failed to process {custom_id} after {num_retries} attempts")


def main():
    args = parse_arguments()
    batch_api_configs = create_batch_api_configs()
    online_AD_with_retries(
        model_name=args.model,
        data_name=args.data,
        request_func=batch_api_configs[args.variant],
        variant=args.variant,
    )


if __name__ == '__main__':
    main()
