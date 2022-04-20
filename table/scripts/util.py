import pandas as pd
import logging
from logging import Formatter, FileHandler, StreamHandler


def get_logger(name, out_dir=None, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not len(logger.handlers):

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler = StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        if out_dir is not None:

            filename = out_dir / f"{name}.log"
            if filename.exists():
                filename.unlink()

            filehandler = FileHandler(filename=filename, encoding="utf-8")
            filehandler.setLevel(level)
            filehandler.setFormatter(formatter)

            logger.addHandler(filehandler)

    return logger


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


def create_or_overwrite(outdir):
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir()
