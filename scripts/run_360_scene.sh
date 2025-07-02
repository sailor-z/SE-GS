dataset=$1
workspace=$2
file=$3
NUM_VIEWS=$4
MAX_ITER=30000

python $file \
--source_path $dataset -m $workspace \
--eval -r 4 --n_views $NUM_VIEWS \
--random_background \
--iterations $MAX_ITER --position_lr_max_steps $MAX_ITER \
--save_iterations $MAX_ITER \
--checkpoint_iterations $MAX_ITER \
--densify_until_iter 15000 \
--densify_grad_threshold 0.0005 \
--gaussiansN 2 \
--coprune --coprune_threshold 20 \
--coreg --sample_pseudo_interval 1 \
--start_sample_pseudo 500 \
--end_sample_pseudo $MAX_ITER \
--noise_ratio_max 0.02 \
--noise_ratio_min 0.001 \


python render.py \
--source_path $dataset -m $workspace \
--render_depth

python metrics.py \
--source_path $dataset -m $workspace \
