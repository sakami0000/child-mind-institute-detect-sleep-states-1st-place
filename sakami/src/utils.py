import functools
import gc
import os
import random
import time
from contextlib import ContextDecorator
from typing import Any, Callable, Type

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def freeze(cls) -> Type:
    """Decorator function for fixing class variables.

    Examples
    --------

        >>> @freeze
        >>> class config:
        >>>     x = 10
        >>>     y = 20

        >>> config.x = 30
        ValueError: Cannot overwrite config.
    """

    class _Const(type):
        """Metaclass of the configuration class.

        Examples
        --------

            >>> class config(metaclass=_Const):
            >>>     x = 10
            >>>     y = 20

            >>> config.x = 30
            ValueError: Cannot overwrite config.

        References
        ----------
        - https://cream-worker.blog.jp/archives/1077207909.html
        """

        def __setattr__(self, name, value):
            raise ValueError("Cannot overwrite config.")

    class frozen_cls(cls, metaclass=_Const):
        pass

    return frozen_cls


class timer(ContextDecorator):
    """Context-manager that logs elapsed time of a process.
    Also functions as a decorator. (Make sure to instantiate with parenthesis.)

    Paramters
    ---------
    message : str
        The displayed message.

    Examples
    --------
    - Usage as a context-manager

        >>> with timer('read csv'):
        >>>     train_df = pd.read_csv(TRAIN_PATH)
        [read csv] start.
        [read csv] done in 0.1 min.

    - Usage as a decorator

        >>> @timer()
        >>> def read_csv():
        >>>     train_df = pd.read_csv(TRAIN_PATH)
        >>>     return train_df
        >>>
        >>> train_df = read_csv()
        [read_csv] start.
        [read_csv] done in 0.1 min.
    """

    def __init__(self, message: str | None = None) -> None:
        self.message = message

    def __call__(self, function: Callable) -> Callable:
        if self.message is None:
            self.message = function.__name__
        return super().__call__(function)

    def __enter__(self) -> None:
        self.start_time = time.time()
        logger.opt(colors=True).info(f"<green>[{self.message}]</green> start.")

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            elapsed_time = time.time() - self.start_time
            logger.opt(colors=True).info(f"<green>[{self.message}]</green> done in {elapsed_time / 60:.1f} min.")


def clear_memory(function: Callable) -> Callable:
    """Decorator function for clearing memory cache."""

    @functools.wraps(function)
    def _clear_memory(*args, **kwargs):
        function(*args, **kwargs)
        gc.collect()
        torch.cuda.empty_cache()

    return _clear_memory
