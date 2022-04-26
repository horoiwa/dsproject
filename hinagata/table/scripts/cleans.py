from pathlib import Path

from .util import save_dataframe, load_dataframe


def clean(input_path: Path, out_path: Path):

    df = load_dataframe(input_path)
    """
    Add your code:
        LGBMが適用できる程度の最低限のクリーニングを行う
    """


    save_dataframe(out_path, df)

    return df
