workspace="SEGS"
file="train_dtu.py"
data_path=$1

num_views=3
bash scripts/run_dtu_scene.sh $data_path/scan8 ./exp/$workspace/dtu_${num_views}/scan8 $file $num_views scan8 True
bash scripts/run_dtu_scene.sh $data_path/scan21 ./exp/$workspace/dtu_${num_views}/scan21 $file $num_views scan21 False
bash scripts/run_dtu_scene.sh $data_path/scan30 ./exp/$workspace/dtu_${num_views}/scan30 $file $num_views scan30 False
bash scripts/run_dtu_scene.sh $data_path/scan31 ./exp/$workspace/dtu_${num_views}/scan31 $file $num_views scan31 False
bash scripts/run_dtu_scene.sh $data_path/scan34 ./exp/$workspace/dtu_${num_views}/scan34 $file $num_views scan34 False
bash scripts/run_dtu_scene.sh $data_path/scan38 ./exp/$workspace/dtu_${num_views}/scan38 $file $num_views scan38 False
bash scripts/run_dtu_scene.sh $data_path/scan40 ./exp/$workspace/dtu_${num_views}/scan40 $file $num_views scan40 True
bash scripts/run_dtu_scene.sh $data_path/scan41 ./exp/$workspace/dtu_${num_views}/scan41 $file $num_views scan41 False
bash scripts/run_dtu_scene.sh $data_path/scan45 ./exp/$workspace/dtu_${num_views}/scan45 $file $num_views scan45 False
bash scripts/run_dtu_scene.sh $data_path/scan55 ./exp/$workspace/dtu_${num_views}/scan55 $file $num_views scan55 False
bash scripts/run_dtu_scene.sh $data_path/scan63 ./exp/$workspace/dtu_${num_views}/scan63 $file $num_views scan63 False
bash scripts/run_dtu_scene.sh $data_path/scan82 ./exp/$workspace/dtu_${num_views}/scan82 $file $num_views scan82 False
bash scripts/run_dtu_scene.sh $data_path/scan103 ./exp/$workspace/dtu_${num_views}/scan103 $file $num_views scan103 False
bash scripts/run_dtu_scene.sh $data_path/scan110 ./exp/$workspace/dtu_${num_views}/scan110 $file $num_views scan110 True
bash scripts/run_dtu_scene.sh $data_path/scan114 ./exp/$workspace/dtu_${num_views}/scan114 $file $num_views scan114 False

num_views=6
bash scripts/run_dtu_scene.sh $data_path/scan8 ./exp/$workspace/dtu_${num_views}/scan8 $file $num_views scan8 False
bash scripts/run_dtu_scene.sh $data_path/scan21 ./exp/$workspace/dtu_${num_views}/scan21 $file $num_views scan21 True
bash scripts/run_dtu_scene.sh $data_path/scan30 ./exp/$workspace/dtu_${num_views}/scan30 $file $num_views scan30 False
bash scripts/run_dtu_scene.sh $data_path/scan31 ./exp/$workspace/dtu_${num_views}/scan31 $file $num_views scan31 False
bash scripts/run_dtu_scene.sh $data_path/scan34 ./exp/$workspace/dtu_${num_views}/scan34 $file $num_views scan34 False
bash scripts/run_dtu_scene.sh $data_path/scan38 ./exp/$workspace/dtu_${num_views}/scan38 $file $num_views scan38 False
bash scripts/run_dtu_scene.sh $data_path/scan40 ./exp/$workspace/dtu_${num_views}/scan40 $file $num_views scan40 False
bash scripts/run_dtu_scene.sh $data_path/scan41 ./exp/$workspace/dtu_${num_views}/scan41 $file $num_views scan41 False
bash scripts/run_dtu_scene.sh $data_path/scan45 ./exp/$workspace/dtu_${num_views}/scan45 $file $num_views scan45 False
bash scripts/run_dtu_scene.sh $data_path/scan55 ./exp/$workspace/dtu_${num_views}/scan55 $file $num_views scan55 False
bash scripts/run_dtu_scene.sh $data_path/scan63 ./exp/$workspace/dtu_${num_views}/scan63 $file $num_views scan63 False
bash scripts/run_dtu_scene.sh $data_path/scan82 ./exp/$workspace/dtu_${num_views}/scan82 $file $num_views scan82 False
bash scripts/run_dtu_scene.sh $data_path/scan103 ./exp/$workspace/dtu_${num_views}/scan103 $file $num_views scan103 False
bash scripts/run_dtu_scene.sh $data_path/scan110 ./exp/$workspace/dtu_${num_views}/scan110 $file $num_views scan110 False
bash scripts/run_dtu_scene.sh $data_path/scan114 ./exp/$workspace/dtu_${num_views}/scan114 $file $num_views scan114 False

num_views=9
bash scripts/run_dtu_scene.sh $data_path/scan8 ./exp/$workspace/dtu_${num_views}/scan8 $file $num_views scan8 False
bash scripts/run_dtu_scene.sh $data_path/scan21 ./exp/$workspace/dtu_${num_views}/scan21 $file $num_views scan21 False
bash scripts/run_dtu_scene.sh $data_path/scan30 ./exp/$workspace/dtu_${num_views}/scan30 $file $num_views scan30 False
bash scripts/run_dtu_scene.sh $data_path/scan31 ./exp/$workspace/dtu_${num_views}/scan31 $file $num_views scan31 False
bash scripts/run_dtu_scene.sh $data_path/scan34 ./exp/$workspace/dtu_${num_views}/scan34 $file $num_views scan34 False
bash scripts/run_dtu_scene.sh $data_path/scan38 ./exp/$workspace/dtu_${num_views}/scan38 $file $num_views scan38 False
bash scripts/run_dtu_scene.sh $data_path/scan40 ./exp/$workspace/dtu_${num_views}/scan40 $file $num_views scan40 False
bash scripts/run_dtu_scene.sh $data_path/scan41 ./exp/$workspace/dtu_${num_views}/scan41 $file $num_views scan41 False
bash scripts/run_dtu_scene.sh $data_path/scan45 ./exp/$workspace/dtu_${num_views}/scan45 $file $num_views scan45 False
bash scripts/run_dtu_scene.sh $data_path/scan55 ./exp/$workspace/dtu_${num_views}/scan55 $file $num_views scan55 False
bash scripts/run_dtu_scene.sh $data_path/scan63 ./exp/$workspace/dtu_${num_views}/scan63 $file $num_views scan63 False
bash scripts/run_dtu_scene.sh $data_path/scan82 ./exp/$workspace/dtu_${num_views}/scan82 $file $num_views scan82 False
bash scripts/run_dtu_scene.sh $data_path/scan103 ./exp/$workspace/dtu_${num_views}/scan103 $file $num_views scan103 False
bash scripts/run_dtu_scene.sh $data_path/scan110 ./exp/$workspace/dtu_${num_views}/scan110 $file $num_views scan110 False
bash scripts/run_dtu_scene.sh $data_path/scan114 ./exp/$workspace/dtu_${num_views}/scan114 $file $num_views scan114 False
