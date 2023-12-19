
python -m components.stacking.main_lgbm \
    feature=stacking_030_truncate \
    lgbm.params.objective=binary \
    lgbm.params.metric=binary_logloss \
    store.model_name=$(basename $0 .sh) \
    lgbm.params.scale_pos_weight=1 \
    lgbm.params.is_unbalance=False

