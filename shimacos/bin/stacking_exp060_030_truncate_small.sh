
for n_fold in 0 1 2 3 4; do
        python -m components.stacking.main_nn \
                base.use_transformer_parameter=False \
                data.n_fold=$n_fold \
                data.is_train=True \
                data.normalize=False \
                data.use_multi_label=True \
                model.model_class=CNNRNNWoDownSample \
                model.num_label=2 \
                base.loss_class=nn.BCEWithLogitsLoss \
                train.warm_start=False \
                store.model_name=$(basename $0 .sh) \
                feature=stacking_030_truncate_small \
                data.dataset_class=StackingDataset \
                train.loss_weight=1 \
                train.batch_size=32 \
                train.epoch=10 \
                train.warmup_multiplier=100 \
                train.learning_rate=0.000001 \
                trainer.precision=16 \
                data.chunk_size=120 \
                data.window_size=60 \
                model.num_layers=1 \
                model.hidden_size=96 \
                model.text.backbone=microsoft/deberta-v3-small \
                model.text.pretrained=False \
                data.use_only_positive_chunk=False \
                data.downsample_feature=False \
                data.is_stacking=True \
                trainer.gradient_clip_val=0.5 \
                base.opt_class=Adam \
                base.distance=8 \
                base.height=0.001 \
                train.refine_target=False \
                train.use_gaussian_target=False \
                trainer.num_sanity_val_steps=5 \
                model.encoder=mean \
                model.decoder=kernel \
                model.kernel_sizes=['3']
done
