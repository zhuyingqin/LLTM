#!/bin/bash
python src/data/synthetic.py --generate --data_dir data/synthetic/range/eval --synthetic_func synthetic_dataset_with_out_of_range_anomalies --seed 42
python src/data/synthetic.py --generate --data_dir data/synthetic/range/train --synthetic_func synthetic_dataset_with_out_of_range_anomalies --seed 3407

python src/data/synthetic.py --generate --data_dir data/synthetic/point/eval --synthetic_func synthetic_dataset_with_point_anomalies --seed 42
python src/data/synthetic.py --generate --data_dir data/synthetic/point/train --synthetic_func synthetic_dataset_with_point_anomalies --seed 3407

python src/data/synthetic.py --generate --data_dir data/synthetic/freq/eval --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 42
python src/data/synthetic.py --generate --data_dir data/synthetic/freq/train --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 3407

python src/data/synthetic.py --generate --data_dir data/synthetic/trend/eval --synthetic_func synthetic_dataset_with_trend_anomalies --seed 42
python src/data/synthetic.py --generate --data_dir data/synthetic/trend/train --synthetic_func synthetic_dataset_with_trend_anomalies --seed 3407

python src/data/synthetic.py --generate --data_dir data/synthetic/flat-trend/eval --synthetic_func synthetic_dataset_with_flat_trend_anomalies --seed 42
python src/data/synthetic.py --generate --data_dir data/synthetic/flat-trend/train --synthetic_func synthetic_dataset_with_flat_trend_anomalies --seed 3407

python src/data/synthetic.py --generate --add_noise --data_dir data/synthetic/noisy-point/eval --synthetic_func synthetic_dataset_with_point_anomalies --seed 42
python src/data/synthetic.py --generate --add_noise --data_dir data/synthetic/noisy-point/train --synthetic_func synthetic_dataset_with_point_anomalies --seed 3407

python src/data/synthetic.py --generate --add_noise --data_dir data/synthetic/noisy-freq/eval --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 42
python src/data/synthetic.py --generate --add_noise --data_dir data/synthetic/noisy-freq/train --synthetic_func synthetic_dataset_with_frequency_anomalies --seed 3407

python src/data/synthetic.py --generate --add_noise --data_dir data/synthetic/noisy-trend/eval --synthetic_func synthetic_dataset_with_trend_anomalies --seed 42
python src/data/synthetic.py --generate --add_noise --data_dir data/synthetic/noisy-trend/train --synthetic_func synthetic_dataset_with_trend_anomalies --seed 3407
