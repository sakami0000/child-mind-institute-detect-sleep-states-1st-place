from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dictionary class that allows for attribute-style access of values."""

    __setattr__ = dict.__setitem__

    def init(self) -> None:
        # update meta_config
        exec_file_name = sys.argv[0]

        self.home_dir = Path("./")
        self.input_dir = Path("../input")
        self.data_dir = self.input_dir / self.competition_name
        self.cache_dir = self.input_dir / "cache"
        self.output_dir = Path("./output")
        self.wandb_dir = Path("./wandb")

        self.run_name = Path(exec_file_name).stem
        self.save_dir = Path(self.output_dir) / self.run_name
        self.checkpoint_dir = self.save_dir / "checkpoint/"

        # make directories
        self.input_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.wandb_dir.mkdir(exist_ok=True)
        self.save_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def check_columns(self) -> None:
        required_columns = [
            "competition_name",
        ]
        for required_column in required_columns:
            if not hasattr(self, required_column):
                raise KeyError(f"Meta config {required_column} must be specified.")

    def __getattr__(self, key: Any) -> Any:
        value = super().get(key)
        if isinstance(value, dict):
            return Config(value)
        return value

    def __deepcopy__(self, memo: dict[int, int | list[int]] | None = None) -> Config:
        """Prevent errors in the `copy.deepcopy` method.

        References
        ----------
        - https://stackoverflow.com/questions/49901590/python-using-copy-deepcopy-on-dotdict
        """
        return Config(copy.deepcopy(dict(self), memo=memo))

    @classmethod
    def load(cls, config_path: str) -> Config:
        """Load a config file.

        Parameters
        ----------
        config_path : str
            Path to config file.

        Returns
        -------
        Config
            Configuration parameters.
        """
        if not Path(config_path).exists():
            raise ValueError(f"Configuration file {config_path} does not exist.")

        with open(config_path) as f:
            config = cls(yaml.safe_load(f))

        config.check_columns()
        config.init()
        return config
