workspace="SEGS"
file="train_llff.py"
data_path=$1

num_views=3
bash scripts/run_llff_scene.sh $data_path/fern ./exp/$workspace/llff_${num_views}/fern $file $num_views
bash scripts/run_llff_scene.sh $data_path/flower ./exp/$workspace/llff_${num_views}/flower $file $num_views
bash scripts/run_llff_scene.sh $data_path/fortress ./exp/$workspace/llff_${num_views}/fortress $file $num_views
bash scripts/run_llff_scene.sh $data_path/horns ./exp/$workspace/llff_${num_views}/horns $file $num_views
bash scripts/run_llff_scene.sh $data_path/leaves ./exp/$workspace/llff_${num_views}/leaves $file $num_views
bash scripts/run_llff_scene.sh $data_path/orchids ./exp/$workspace/llff_${num_views}/orchids $file $num_views
bash scripts/run_llff_scene.sh $data_path/room ./exp/$workspace/llff_${num_views}/room $file $num_views
bash scripts/run_llff_scene.sh $data_path/trex ./exp/$workspace/llff_${num_views}/trex $file $num_views

num_views=6
bash scripts/run_llff_scene.sh $data_path/fern ./exp/$workspace/llff_${num_views}/fern $file $num_views
bash scripts/run_llff_scene.sh $data_path/flower ./exp/$workspace/llff_${num_views}/flower $file $num_views
bash scripts/run_llff_scene.sh $data_path/fortress ./exp/$workspace/llff_${num_views}/fortress $file $num_views
bash scripts/run_llff_scene.sh $data_path/horns ./exp/$workspace/llff_${num_views}/horns $file $num_views
bash scripts/run_llff_scene.sh $data_path/leaves ./exp/$workspace/llff_${num_views}/leaves $file $num_views
bash scripts/run_llff_scene.sh $data_path/orchids ./exp/$workspace/llff_${num_views}/orchids $file $num_views
bash scripts/run_llff_scene.sh $data_path/room ./exp/$workspace/llff_${num_views}/room $file $num_views
bash scripts/run_llff_scene.sh $data_path/trex ./exp/$workspace/llff_${num_views}/trex $file $num_views

num_views=9
bash scripts/run_llff_scene.sh $data_path/fern ./exp/$workspace/llff_${num_views}/fern $file $num_views
bash scripts/run_llff_scene.sh $data_path/flower ./exp/$workspace/llff_${num_views}/flower $file $num_views
bash scripts/run_llff_scene.sh $data_path/fortress ./exp/$workspace/llff_${num_views}/fortress $file $num_views
bash scripts/run_llff_scene.sh $data_path/horns ./exp/$workspace/llff_${num_views}/horns $file $num_views
bash scripts/run_llff_scene.sh $data_path/leaves ./exp/$workspace/llff_${num_views}/leaves $file $num_views
bash scripts/run_llff_scene.sh $data_path/orchids ./exp/$workspace/llff_${num_views}/orchids $file $num_views
bash scripts/run_llff_scene.sh $data_path/room ./exp/$workspace/llff_${num_views}/room $file $num_views
bash scripts/run_llff_scene.sh $data_path/trex ./exp/$workspace/llff_${num_views}/trex $file $num_views
