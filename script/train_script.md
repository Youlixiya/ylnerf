# training script
```shell
#fern
python train.py --root_dir data/nerf_llff_data/flower --dataset_name llff --img_wh 756, 1008 --batch_size 1024 --num_epochs 16 --exp_name flower --workers 16 --use_wandb --project_name nerf_flower
```