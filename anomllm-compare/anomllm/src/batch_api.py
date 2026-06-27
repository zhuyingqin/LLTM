import json
import argparse
from utils import generate_batch_AD_requests
from openai_api import openai_client
from loguru import logger
import json
import os
from config import create_batch_api_configs


def check_existing_batch(client, batch_id):
    try:
        batch = client.batches.retrieve(batch_id)
        return batch
    except Exception as e:
        logger.error(f"Error retrieving batch: {e}")
        return None


def generate_and_save_batch(client, variant, batch_api_configs, model_name, data_name):
    jsonl_fn = generate_batch_AD_requests(
        model_name=model_name,
        data_name=data_name,
        request_func=batch_api_configs[variant],
        variant=variant
    )
    batch_input_file = client.files.create(
        file=open(jsonl_fn, "rb"),
        purpose="batch"
    )
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    return batch


def save_batch_to_file(batch, batch_key, filename):
    try:
        with open(filename, 'r') as f:
            existing_batches = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_batches = {}

    existing_batches[batch_key] = batch

    with open(filename, 'w') as f:
        json.dump(existing_batches, f, default=lambda obj: obj.__dict__, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process batch generation options.')
    parser.add_argument('--variant', type=str, default='1shot-vision', help='Variant type')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model name')
    parser.add_argument('--data', type=str, default='point', help='Data name')
    return parser.parse_args()


def retreive_result(client, batch):
    input_file_content = client.files.content(batch.input_file_id)
    output_file_content = client.files.content(batch.output_file_id)
    output_json = [json.loads(line) for line in output_file_content.text.strip().split('\n')]
    input_json = [json.loads(line) for line in input_file_content.text.strip().split('\n')]
    
    # Match and dump
    result_jsonl = []
    for input_line, output_line in zip(input_json, output_json):
        assert input_line['custom_id'] == output_line['custom_id']
        result_jsonl.append({
            "custom_id": input_line['custom_id'],
            "request": input_line['body'],
            "response": output_line['response']['body']['choices'][0]['message']['content'] 
        })
    return result_jsonl
        

def main():
    args = parse_arguments()
    batch_api_configs = create_batch_api_configs()
    client = openai_client(args.model)

    batch_key = f'{args.data}_{args.model}_{args.variant}'
    result_fn = f"results/synthetic/{args.data}/{args.model}/{args.variant}.jsonl"

    # Check if batch exists
    batch_fn = f'results/synthetic/{args.data}/{args.model}/{args.variant}_batch.json'
    try:
        with open(batch_fn, 'r') as f:
            existing_batches = json.load(f)
            if batch_key in existing_batches:
                logger.info(f"Existing batch for {batch_key} found: {existing_batches[batch_key]['id']}")
                status = existing_batches[batch_key]['status']
                batch = check_existing_batch(client, existing_batches[batch_key]['id'])
                logger.debug(f"Batch {existing_batches[batch_key]['id']} status: {status} -> {batch.status}")
                if batch.status == 'completed':
                    logger.debug(f"Batch {existing_batches[batch_key]['id']} is completed")
                    if not os.path.exists(result_fn):
                        # Retrieve the batch
                        result = retreive_result(client, batch)
                        with open(result_fn, 'w') as outfile:
                            for item in result:
                                outfile.write(json.dumps(item) + '\n')
                        logger.info(f"Batch {existing_batches[batch_key]['id']} result saved to {result_fn}")
                    else:
                        logger.debug(f"Batch {existing_batches[batch_key]['id']} result already saved, do nothing")
                else:
                    logger.debug(f"Batch {existing_batches[batch_key]['id']} is still wait in progress")
                if batch:
                    save_batch_to_file(batch, batch_key, batch_fn)
                    return
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        logger.error(f"Error loading existing batch: {e}")

    # If not exists, generate a new batch
    logger.info(f"Generating new batch for {batch_key}...")
    batch = generate_and_save_batch(client, args.variant, batch_api_configs, args.model, args.data)
    save_batch_to_file(batch, batch_key, batch_fn)


if __name__ == '__main__':
    main()
