data_id=8

python dual_scripts/sample_for_pocket/sample_for_pocket.py \
    configs/sampling.yml \
    --data_id ${data_id} \
    --result_path outputs/baseline_targetdiff/${data_id}