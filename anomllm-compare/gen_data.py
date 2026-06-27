"""Generate AnomLLM's synthetic anomaly datasets locally, using THEIR exact generator
functions (src/data/synthetic.py) but bypassing their utils->openai_api import chain
(we only need the data, not the LLM plotting). Same anomaly funcs, same seeds as
synthesize.sh (eval=42, train=3407). Each series is univariate (number_of_sensors=1),
matching dataset.generate()."""
import argparse
import importlib.util
import os
import pickle
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SYN_PATH = os.path.join(ROOT, "anomllm", "src", "data", "synthetic.py")

# stub `utils` so importing synthetic.py does NOT pull in openai_api/loguru/etc.
_stub = types.ModuleType("utils")
_stub.plot_series_and_predictions = lambda **k: None
sys.modules["utils"] = _stub

spec = importlib.util.spec_from_file_location("anom_synthetic", SYN_PATH)
syn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(syn)

FUNCS = {
    "point": "synthetic_dataset_with_point_anomalies",
    "range": "synthetic_dataset_with_out_of_range_anomalies",
    "freq":  "synthetic_dataset_with_frequency_anomalies",
    "trend": "synthetic_dataset_with_trend_anomalies",
}


def gen_one(func_name, n, seed, add_noise):
    np.random.seed(seed)
    func = getattr(syn, func_name)
    series, anom = [], []
    for _ in range(n):
        data, loc = func(number_of_sensors=1, ratio_of_anomalous_sensors=1.0)
        if add_noise:
            data = data + np.random.normal(0, 0.08, data.shape)
        series.append(np.asarray(data, dtype=np.float32))
        anom.append(loc)
    return {"series": series, "anom": anom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--types", default="point,range,trend,freq")
    ap.add_argument("--num-series", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--noise", action="store_true")
    args = ap.parse_args()

    for t in args.types.split(","):
        for split, seed in (("eval", 42), ("train", 3407)):
            d = gen_one(FUNCS[t], args.num_series, seed, args.noise)
            out_dir = os.path.join(args.out, t, split)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "data.pkl"), "wb") as f:
                pickle.dump(d, f)
            n_anom = sum(len(a[0]) for a in d["anom"])
            print(f"{t}/{split}: {len(d['series'])} series, shape {d['series'][0].shape}, "
                  f"{n_anom} anomaly intervals -> {out_dir}/data.pkl")


if __name__ == "__main__":
    main()
