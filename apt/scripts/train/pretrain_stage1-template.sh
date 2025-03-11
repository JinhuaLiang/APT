export OMP_NUM_THREADS=1

conda activate /path/to/lam
nvidia-smi

export WORK_PLACE=/path/to/APT/apt/src/lavis


python -m torch.distributed.run \
    --nproc_per_node=1 ${WORK_PLACE}/train.py \
    --cfg-path ${WORK_PLACE}/lavis/projects/lam/train/pretrain_stage1.yaml \
    --options \
    run.max_iters=98000 \
    run.iters_per_inner_epoch=6250 \
    run.batch_size_train=400 \
    run.num_workers=4 \
    run.output_dir="/path/to/output_dir"