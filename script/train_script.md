# training script
```shell
#fern
python train.py --root_dir data/nerf_llff_data/flower --dataset_name llff --img_wh 756 1008 --batch_size 1024 --num_epochs 16 --exp_name flower --workers 16 --use_wandb --project_name nerf_flower
```
```shell
#lego
python3 train.py --root_dir data/nerf_synthetic/lego --dataset_name blender --img_wh 400 400 --batch_size 4096 --num_epochs 10 --exp_name lego --workers 16 --use_wandb --project_name nerf_lego --devices 2
```
# eval script
```shell
python3 eval.py --root_dir data/nerf_synthetic/lego --exp_name lego --dataset_name blender --ckpt_path lego/ckpts/epoch=2-val_mean_psnr=19.602102279663086.ckpt --img_wh 400 400
```
