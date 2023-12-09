FROM gcr.io/kaggle-gpu-images/python:v139

RUN pip install \
    hydra-core==1.3.2 \
    segmentation_models_pytorch==0.3.3 \
    google_cloud_storage==2.9.0 \
    invoke==2.2.0 \
    loguru==0.7.0 \
    nbparameterise==0.6 \
    lightgbm==4.0.0
