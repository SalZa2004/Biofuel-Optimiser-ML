import pandas as pd

def read_csv_safely(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"Could not read CSV file {path}. "
        "Tried utf-8, latin-1, iso-8859-1, cp1252."
    )


def format_mixture_dataset(input_csv: str) -> pd.DataFrame:
    df = read_csv_safely(input_csv)

    # (rest of your formatting logic here)
    return df

