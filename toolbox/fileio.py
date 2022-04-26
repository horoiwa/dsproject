from pathlib import Path
import json
import logging
from logging import Formatter, FileHandler, StreamHandler


def save_dict_as_json(d, out_path):
    for key in d.keys():
        d[key] = str(d[key])
    with open(out_path, 'w') as f:
        json.dump(d, f, indent=2, ensure_ascii=False)


def get_logger(name="root", logfile=None, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not len(logger.handlers):

        formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler = StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        if logfile is not None:

            filename = Path(logfile)
            if filename.exists():
                filename.unlink()

            filehandler = FileHandler(filename=filename, encoding="utf-8")
            filehandler.setLevel(level)
            filehandler.setFormatter(formatter)

            logger.addHandler(filehandler)

    return logger
