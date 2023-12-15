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
