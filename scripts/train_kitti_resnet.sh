CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ktnet_kitti_R_101.yaml \
    OUTPUT_DIR work_dir/da_ktnet_kitti_R_16
