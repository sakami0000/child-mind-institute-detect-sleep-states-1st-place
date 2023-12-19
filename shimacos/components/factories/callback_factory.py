import os
from glob import glob

from google.cloud import storage

from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar


class MyCallback(ModelCheckpoint):
    def __init__(
        self,
        store_config,
        monitor="val_loss",
        verbose=0,
        save_top_k=1,
        save_weights_only=False,
        mode="auto",
        period=1,
    ):
        super(MyCallback, self).__init__(
            store_config.model_path,
            monitor,
            verbose,
            save_top_k,
            save_weights_only,
            mode,
            period,
            store_config.model_class,
        )
        self.store_config = store_config

    def _save_model(self, filepath):
        dirpath = os.path.dirname(filepath)
        # make paths
        os.makedirs(dirpath, exist_ok=True)

        # delegate the saving to the model
        self.save_function(filepath)
        if self.store_config.gcs_project is not None:
            self.upload_directory()

    def upload_directory(self):
        storage_client = storage.Client(self.store_config.gcs_project)
        bucket = storage_client.get_bucket(self.store_config.bucket_name)
        filenames = glob(
            os.path.join(self.store_config.save_path, "**"), recursive=True
        )
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            destination_blob_name = os.path.join(
                self.store_config.gcs_path,
                filename.split(self.store_config.save_path)[-1][1:],
            )
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(filename)


class MyProgressBar(ProgressBar):
    def format_num(self, n):
        f = "{0:2.3g}".format(n)
        f = f.replace("+0", "+")
        f = f.replace("-0", "-")
        n = str(n)
        return f if len(f) < len(n) else n

    def on_batch_end(self, trainer, pl_module):
        # super().on_batch_end(trainer, pl_module)
        if self.is_enabled and self.train_batch_idx % self.refresh_rate == 0:
            self.main_progress_bar.update(self.refresh_rate)
            for key, val in trainer.progress_bar_dict.items():
                if not isinstance(val, str):
                    trainer.progress_bar_dict[key] = self.format_num(val)
            self.main_progress_bar.set_postfix(trainer.progress_bar_dict)

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        for key, val in trainer.progress_bar_dict.items():
            if not isinstance(val, str):
                trainer.progress_bar_dict[key] = self.format_num(val)
        self.main_progress_bar.set_postfix(trainer.progress_bar_dict)
        self.val_progress_bar.close()
