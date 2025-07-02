workspace="SEGS"
file="train_360.py"
data_path=$1

num_views=12
bash scripts/run_360_scene.sh $data_path/bicycle ./exp/$workspace/mipnerf360_${num_views}/bicycle $file $num_views
bash scripts/run_360_scene.sh $data_path/bonsai ./exp/$workspace/mipnerf360_${num_views}/bonsai $file $num_views
bash scripts/run_360_scene.sh $data_path/counter ./exp/$workspace/mipnerf360_${num_views}/counter $file $num_views
bash scripts/run_360_scene.sh $data_path/garden ./exp/$workspace/mipnerf360_${num_views}/garden $file $num_views
bash scripts/run_360_scene.sh $data_path/kitchen ./exp/$workspace/mipnerf360_${num_views}/kitchen $file $num_views
bash scripts/run_360_scene.sh $data_path/room ./exp/$workspace/mipnerf360_${num_views}/room $file $num_views
bash scripts/run_360_scene.sh $data_path/stump ./exp/$workspace/mipnerf360_${num_views}/stump $file $num_views


num_views=24
bash scripts/run_360_scene.sh $data_path/bicycle ./exp/$workspace/mipnerf360_${num_views}/bicycle $file $num_views
bash scripts/run_360_scene.sh $data_path/bonsai ./exp/$workspace/mipnerf360_${num_views}/bonsai $file $num_views
bash scripts/run_360_scene.sh $data_path/counter ./exp/$workspace/mipnerf360_${num_views}/counter $file $num_views
bash scripts/run_360_scene.sh $data_path/garden ./exp/$workspace/mipnerf360_${num_views}/garden $file $num_views
bash scripts/run_360_scene.sh $data_path/kitchen ./exp/$workspace/mipnerf360_${num_views}/kitchen $file $num_views
bash scripts/run_360_scene.sh $data_path/room ./exp/$workspace/mipnerf360_${num_views}/room $file $num_views
bash scripts/run_360_scene.sh $data_path/stump ./exp/$workspace/mipnerf360_${num_views}/stump $file $num_views
