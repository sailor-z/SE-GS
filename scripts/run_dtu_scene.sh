dataset=$1
workspace=$2
file=$3
NUM_VIEWS=$4
SCAN_ID=$5
RAND_PCD=$6
MAX_ITER=10000

if [ "$RAND_PCD" == "True" ]
then
  python $file \
  --source_path $dataset -m $workspace \
  --eval  -r 4 --n_views $NUM_VIEWS \
  --iterations $MAX_ITER --position_lr_max_steps $MAX_ITER \
  --densify_until_iter $MAX_ITER \
  --densify_grad_threshold 0.0005 \
  --gaussiansN 2 \
  --coprune --coprune_threshold 10 \
  --coreg \
  --sample_pseudo_interval 1 \
  --start_sample_pseudo 2000 \
  --rand_pcd \
  --position_lr_init 0.00016 \
  --position_lr_final 0.00016 \
  --noise_ratio_max 0.08 \
  --noise_ratio_min 0.02 \

else
  python $file \
  --source_path $dataset -m $workspace \
  --eval  -r 4 --n_views $NUM_VIEWS \
  --iterations $MAX_ITER --position_lr_max_steps $MAX_ITER \
  --densify_until_iter $MAX_ITER \
  --densify_grad_threshold 0.0005 \
  --gaussiansN 2 \
  --coprune --coprune_threshold 10 \
  --coreg \
  --sample_pseudo_interval 1 \
  --start_sample_pseudo 2000 \
  --position_lr_init 0.00016 \
  --position_lr_final 0.00016 \
  --noise_ratio_max 0.08 \
  --noise_ratio_min 0.02 \

fi

bash ./copy_mask_dtu.sh $workspace $SCAN_ID

python render.py \
--source_path $dataset -m $workspace \

python metrics_dtu.py \
-m $workspace \
