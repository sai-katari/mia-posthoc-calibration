import logging
import os
import sys


def setup_logger(run_id: str, log_dir: str = "experiments") -> logging.Logger:
    os.makedirs(f"{log_dir}/{run_id}", exist_ok=True)
    log_path = f"{log_dir}/{run_id}/train.log"

    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger  # already configured (e.g. during tests)

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
