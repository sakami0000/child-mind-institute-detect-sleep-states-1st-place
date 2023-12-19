import os
import random
import re
import shutil
from glob import glob

import components.stacking.runner as runner
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True, warn_only=True)


def prepair_dir(config: DictConfig):
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
        config.store.feature_path,
    ]:
        if os.path.exists(path) and config.train.warm_start is False and config.data.is_train:
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_up(config: DictConfig):
    # Setup
    prepair_dir(config)
    set_seed(config.train.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


@hydra.main(config_path="../../yamls", config_name="nn.yaml")
def main(config: DictConfig):
    os.chdir(config.store.workdir)
    set_up(config)
    if config.train.warm_start:
        checkpoint_path = sorted(
            glob(config.store.model_path + "/*epoch*"),
            key=lambda path: int(re.split("[=.]", path)[-2]),
        )[-1]
        print(checkpoint_path)
    else:
        checkpoint_path = None

    model = getattr(runner, config.runner)(config)

    if config.data.is_train:
        trainer = instantiate(config.trainer)
        trainer.fit(model)
    else:
        state_dict = torch.load(checkpoint_path)["state_dict"]
        model.load_state_dict(state_dict)
        config.trainer.update(
            {
                "devices": 1,
                "logger": None,
                "limit_train_batches": 0.0,
                "limit_val_batches": 0.0,
                "limit_test_batches": 1.0,
                "accelerator": None,
            }
        )
        trainer = instantiate(config.trainer)
        trainer.test(model, model.test_dataloader())


if __name__ == "__main__":
    main()
