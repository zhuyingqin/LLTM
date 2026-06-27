import argparse
import numpy as np
import pandas as pd
from tqdm import trange
from prompt import time_series_to_image
from utils import (
    view_base64_image,
    display_messages,
    collect_results,
    plot_series_and_predictions,
    interval_to_vector,
    compute_metrics,
    process_dataframe,
    highlight_by_ranking,
    styled_df_to_latex,
)
import pickle
import os
from data.synthetic import SyntheticDataset


def load_datasets(data_name):
    data_dir = f"data/synthetic/{data_name}/eval/"
    train_dir = f"data/synthetic/{data_name}/train/"
    eval_dataset = SyntheticDataset(data_dir)
    eval_dataset.load()
    train_dataset = SyntheticDataset(train_dir)
    train_dataset.load()
    return eval_dataset, train_dataset


def compute_metrics_for_results(eval_dataset, results, num_samples=400):
    metric_names = [
        "precision",
        "recall",
        "f1",
        "affi precision",
        "affi recall",
        "affi f1",
    ]
    results_dict = {key: [[] for _ in metric_names] for key in results.keys()}

    for i in trange(0, num_samples):
        anomaly_locations = eval_dataset[i][0].numpy()
        gt = interval_to_vector(anomaly_locations[0])
        
        for name, prediction in results.items():
            try:
                metrics = compute_metrics(gt, prediction[i])
            except IndexError:
                print(f"experiment {name} not finished")
            for idx, metric_name in enumerate(metric_names):
                results_dict[name][idx].append(metrics[metric_name])

    df = pd.DataFrame(
        {k: np.mean(v, axis=1) for k, v in results_dict.items()},
        index=["precision", "recall", "f1", "affi precision", "affi recall", "affi f1"],
    )
    return df


def main(args):
    data_name = args.data_name
    label_name = args.label_name
    table_caption = args.table_caption
    
    # Load results if already computed
    if False:
        # if os.path.exists(f"results/agg/{data_name}.pkl"):
        with open(f"results/agg/{data_name}.pkl", "rb") as f:
            double_df = pickle.load(f)   
    else:
        eval_dataset, train_dataset = load_datasets(data_name)
        directory = f"results/synthetic/{data_name}"
        results = collect_results(directory, ignore=['phi'])
        df = compute_metrics_for_results(eval_dataset, results)
        double_df = process_dataframe(df.T.copy())
        print(double_df)
        
        # Saved results double_df to pickle
        with open(f"results/agg/{data_name}.pkl", "wb") as f:
            pickle.dump(double_df, f)
        
    styled_df = highlight_by_ranking(double_df)

    latex_table = styled_df_to_latex(styled_df, table_caption, label=label_name)
    print(latex_table)
    
    # Also append the table to out.tex
    with open("out.tex", "a") as f:
        f.write(latex_table)


"""
python src/result_agg.py --data_name trend --label_name trend-exp --table_caption "Trend anomalies in shifting sine wave"
python src/result_agg.py --data_name freq --label_name freq-exp --table_caption "Frequency anomalies in regular sine wave"
python src/result_agg.py --data_name point --label_name point-exp --table_caption "Point noises anomalies in regular sine wave"
python src/result_agg.py --data_name range --label_name range-exp --table_caption "Out-of-range anomalies in Gaussian noise"

python src/result_agg.py --data_name noisy-trend --label_name noisy-trend-exp --table_caption "Trend anomalies in shifting sine wave with extra noise"
python src/result_agg.py --data_name noisy-freq --label_name noisy-freq-exp --table_caption "Frequency anomalies in regular sine wave with extra noise"
python src/result_agg.py --data_name noisy-point --label_name noisy-point-exp --table_caption "Point noises anomalies in regular sine wave with Gaussian noise"
python src/result_agg.py --data_name flat-trend --label_name flat-trend-exp --table_caption "Trend anomalies, but no negating trend, and less noticeable speed changes"
"""  # noqa

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process time series data and generate LaTeX table."
    )
    parser.add_argument("--data_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--label_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--table_caption", type=str, required=True, help="Caption for the LaTeX table")
    args = parser.parse_args()
    main(args)
