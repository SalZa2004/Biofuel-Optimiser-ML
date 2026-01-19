import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", default="results/")

    return parser.parse_args()
