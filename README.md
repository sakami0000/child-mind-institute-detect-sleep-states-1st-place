## Prepare dataset

- download from kaggle via kaggle-api.
  ```bash
  kaggle competitions download -c child-mind-institute-detect-sleep-states -p ./input/
  unzip ./input/child-mind-institute-detect-sleep-states.zip -d ./input
  ```

## How to train

- setting up using Docker

  ```sh
  docker compose up -d
  docker compose exec kaggle /bin/bash
  ```

- training all
  ```sh
  inv run-all
  ```
  - (Optional) If you want to re-split folds, add `--overwrite-folds` option.
    - Not reproducible, so results change every time it is run.
    - Default values exists in input/folds which we used
    ```sh
    inv run-all --overwrite-folds
    ```

## How to submission

- Use [this notebook](https://www.kaggle.com/shimacos/kaggle-smi-submission-final)
