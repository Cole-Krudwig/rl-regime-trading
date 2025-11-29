# cli.py

from src.Fetch import Fetch
from src.evaluate_benchmarks import BenchmarkEvaluator
from src.Evaluate import ModelEvaluator
from src.Train import Train
import argparse
import os
import sys

# --- Set up paths so we can import src modules cleanly ---

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")

'''if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)'''

# Now these match your project structure and internal imports

DATA_DIR = os.path.join(ROOT_DIR, "data")


def ensure_data(symbol: str) -> str:
    """
    Ensure that a CSV for the given symbol exists under data/.
    If not, use Fetch to download it via Alpha Vantage.

    Returns:
        Absolute path to data/<symbol>.csv
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    filename = f"{symbol}.csv"
    data_path = os.path.join(DATA_DIR, filename)

    if os.path.exists(data_path):
        print(f"[INFO] Found existing data file for {symbol}: {data_path}")
        return data_path

    print(
        f"[INFO] Data file for {symbol} not found. Fetching with Fetch class...")

    fetcher = Fetch()

    # IMPORTANT:
    # Fetch.fetch currently saves to '../data/<symbol>.csv' relative to its
    # current working directory. To make that resolve to ROOT_DIR/data,
    # we temporarily chdir into SRC_DIR before calling it.
    old_cwd = os.getcwd()
    try:
        os.chdir(SRC_DIR)
        # uses get_daily + json_to_df + Log_Return + to_csv('../data/...')
        df = fetcher.fetch(symbol)
    finally:
        os.chdir(old_cwd)

    # After this, Fetch has written ROOT_DIR/data/<symbol>.csv.
    # As a safety net, if for some reason it's not there, save df ourselves.
    if not os.path.exists(data_path):
        print(
            "[WARN] Expected fetched file not found at "
            f"{data_path}, writing CSV directly."
        )
        df.to_csv(data_path)

    print(f"[INFO] Saved fetched data for {symbol} to {data_path}")
    return data_path


# ----------------- Command handlers ----------------- #

def run_train(args: argparse.Namespace) -> None:
    """
    Train all RL models (DQN, PPO, A2C) for the given symbol.
    """
    symbol = args.symbol
    train_split = args.train_split
    timesteps = args.timesteps

    data_path = ensure_data(symbol)

    print(
        f"\n[TRAIN] Symbol={symbol}, data={data_path}, "
        f"train_split={train_split}, timesteps={timesteps}"
    )

    trainer = Train(data_path=data_path, train_split=train_split)
    trainer.train_all(total_timesteps=timesteps)


def run_evaluate(args: argparse.Namespace) -> None:
    """
    Evaluate trained RL models and benchmarks on the final (1 - train_split)
    portion of the data.
    """
    symbol = args.symbol
    train_split = args.train_split

    # Use explicit data path if given, otherwise infer from symbol
    if args.data is not None:
        data_path = os.path.abspath(args.data)
        print(f"[EVAL] Using explicit data path: {data_path}")
    else:
        data_path = ensure_data(symbol)
        print(f"[EVAL] Using inferred data path for {symbol}: {data_path}")

    print(
        f"\n[EVAL] Symbol={symbol}, data={data_path}, "
        f"train_split={train_split} "
        f"(evaluate on final {100 * (1 - train_split):.1f}% of data)"
    )

    # Evaluate RL models: DQN, PPO, A2C
    model_evaluator = ModelEvaluator(
        data_path=data_path, train_split=train_split)
    model_evaluator.evaluate_all()  # prints the RL models summary table

    # Evaluate benchmarks: 100% risky, 60/40, base strategies
    benchmark_evaluator = BenchmarkEvaluator(
        data_path=data_path,
        train_split=train_split,
    )
    benchmark_evaluator.evaluate_benchmarks()  # prints benchmark summary


# ----------------- CLI parser ----------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RL Regime Trading - Training and Evaluation CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- train command ----
    train_parser = subparsers.add_parser(
        "train", help="Train DQN/PPO/A2C models for a given symbol"
    )
    train_parser.add_argument(
        "--symbol",
        required=True,
        help="Ticker symbol to train on (e.g. SPY)",
    )
    train_parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help=(
            "Fraction of data used for training. "
            "The final (1 - train_split) fraction is left out for evaluation. "
            "Example: 0.8 -> final 20%% reserved for test. (default: 0.8)"
        ),
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps for each RL model (default: 100000)",
    )
    train_parser.set_defaults(func=run_train)

    # ---- evaluate command ----
    eval_parser = subparsers.add_parser(
        "evaluate",
        help=(
            "Evaluate trained models and benchmarks on the final portion of the data "
            "(defined by 1 - train_split)."
        ),
    )
    eval_parser.add_argument(
        "--symbol",
        required=True,
        help="Ticker symbol to evaluate (e.g. SPY)",
    )
    eval_parser.add_argument(
        "--data",
        default=None,
        help=(
            "Optional explicit path to CSV data file. "
            "If omitted, uses data/<symbol>.csv, fetching via Fetch if needed."
        ),
    )
    eval_parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help=(
            "Fraction of data treated as 'train'; the final (1 - train_split) is "
            "the evaluation window. Use the same value as during training "
            "for consistency. (default: 0.8)"
        ),
    )
    eval_parser.set_defaults(func=run_evaluate)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
