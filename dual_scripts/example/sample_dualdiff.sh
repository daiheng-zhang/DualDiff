synergy_idx=10347

python dual_scripts/sample_for_pocket/compose_sample_score.py \
        configs/sampling.yml \
        --synergy_idx ${synergy_idx} \
        --num_samples 20 \
        --result_path outputs/composition_score_sample