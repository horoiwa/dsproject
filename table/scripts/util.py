import pandas as pd


def save_dataframe(out_path, df):

    suffix = out_path.suffix[1:]
    if suffix == "csv":
        df.to_csv(out_path, index=None)
    elif suffix == "hdf":
        df.to_hdf(out_path, key='df', mode='w')
    else:
        raise NotImplementedError(f"Not implemented: {suffix}")


def load_dataframe(input_path) -> pd.DataFrame:

    suffix = input_path.suffix[1:]
    if suffix == "csv":
        df = pd.read_csv(input_path)
    elif suffix == "hdf":
        df = pd.read_hdf(input_path)
    else:
        raise NotImplementedError(f"Not implemented: {suffix}")

    return df
