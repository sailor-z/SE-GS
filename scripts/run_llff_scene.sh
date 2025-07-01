dataset=$1
workspace=$2
file=$3
MAX_ITER=10000
NUM_VIEWS=$4

python $file \
--source_path $dataset -m $workspace \
--eval -r 8 --n_views $NUM_VIEWS \
--random_background \
--iterations $MAX_ITER --position_lr_max_steps $MAX_ITER \
--gaussiansN 2 \
--coprune --coprune_threshold 5 \
--coreg --start_sample_pseudo 500 \
--densify_grad_threshold 0.0005 \
--densify_until_iter $MAX_ITER \
--end_sample_pseudo $MAX_ITER \
--sample_pseudo_interval 1 \
--position_lr_init 0.00016 \
--position_lr_final 0.00016 \
--noise_ratio_max 0.08 \
--noise_ratio_min 0.02 \

python render.py \
--source_path $dataset -m $workspace \
--render_depth

python metrics.py \
--source_path $dataset -m $workspace \
