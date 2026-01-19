from applications.mixture_predictor.cli import parse_args
from applications.mixture_predictor.results import display_results

from core.predictors.mixture.pipeline import run_mixture_cn_pipeline


def main():
    args = parse_args()
    df, stats = run_mixture_cn_pipeline(args)
    display_results(df, stats)


if __name__ == "__main__":
    main()
