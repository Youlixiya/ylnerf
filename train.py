import os
from opt import get_opts
from collections import defaultdict
from PIL import Image
from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *
from torch.optim import Adam, AdamW, RAdam, SGD
import wandb
from transformers import get_cosine_schedule_with_warmup

# losses
from losses import loss_dict

# metrics
from metrics import *
from torchmetrics import MeanMetric
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import WandbLogger


class NeRFSystem(LightningModule):

    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.save_hyperparameters(hparams)

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]
        self.train_loss = MeanMetric()
        self.valid_loss = MeanMetric()
        self.train_psnr = MeanMetric()
        self.valid_psnr= MeanMetric()
        self.automatic_optimization = False
    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    # def prepare_data(self):
    #
    def setup(self, stage: str) -> None:
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                       'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.devices
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = eval(self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr,weight_decay = self.hparams.weight_decay)
        lr_warmup_steps = len(self.train_dataloader()) * self.hparams.warmup_epochs
        self.train_total_steps = len(self.train_dataloader()) * self.hparams.num_epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=self.train_total_steps)
        return [self.optimizer], [self.lr_scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.workers,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        self.log('lr', lr_scheduler.get_last_lr()[0], prog_bar=True, on_step=True, on_epoch=False, logger=True if self.hparams.use_wandb else False)
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        loss = self.loss(results, rgbs)
        self.train_loss.update(loss)
        self.log('train_loss', self.train_loss(loss), prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.hparams.use_wandb else False)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        optimizer.zero_grad()
        self.manual_backward(loss)
        # 调用优化器根据梯度更新模型参数
        optimizer.step()
        # 更新学习率，每训练一个批次学习率都会变化，学习率先会经过热身阶段从很小的值变成初始设置的值然后学习率会不断下降
        lr_scheduler.step()

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            self.train_psnr.update(psnr_)
            # log['train/psnr'] = psnr_
            self.log('train_psnr', psnr_, prog_bar=True, on_step=True, on_epoch=False,
                     logger=True if self.hparams.use_wandb else False)

        return loss
    def on_train_epoch_end(self) -> None:
        mean_loss = self.train_loss.compute()
        mean_psnr = self.train_psnr.compute()
        self.train_loss.reset()
        self.train_psnr.reset()
        self.log('train_mean_loss', mean_loss, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.hparams.use_wandb else False)
        self.log('train_mean_psnr', mean_psnr, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.hparams.use_wandb else False)
    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        valid_loss = self.loss(results, rgbs)
        self.valid_loss.update(valid_loss)
        self.log('val_loss', valid_loss, prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.hparams.use_wandb else False)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            samples_dir = os.path.join(self.hparams.exp_name, 'samples')
            rgb_samples_dir = os.path.join(samples_dir, 'results_rgb')
            deep_samples_dir = os.path.join(samples_dir, 'results_depth')
            if not os.path.exists(samples_dir): os.makedirs(samples_dir)
            if not os.path.exists(rgb_samples_dir): os.makedirs(rgb_samples_dir)
            if not os.path.exists(deep_samples_dir): os.makedirs(deep_samples_dir)
            H, W = self.hparams.img_wh
            img = numpy_to_pil((results[f'rgb_{typ}']).view(H, W, 3).clamp(0, 1).cpu().numpy())[0]
            img_gt = numpy_to_pil((rgbs.view(H, W, 3)).clamp(0, 1).cpu().numpy())[0]
            depth = numpy_to_pil((visualize_depth(results[f'depth_{typ}'].view(H, W))).permute(1, 2, 0).clamp(0, 1).cpu().numpy())[0]
            img.save(os.path.join(rgb_samples_dir, f'rgb_{typ}_{self.trainer.current_epoch}.png'))
            img_gt.save(os.path.join(samples_dir, 'img_gt.png'))
            depth.save(os.path.join(deep_samples_dir, f'depth_{typ}_{self.trainer.current_epoch}.png'))
            if self.hparams.use_wandb:
                wandb.log({'img_gt' : wandb.Image(img_gt),
                           'img' : wandb.Image(img),
                           'depth' : wandb.Image(depth)})

        # log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        valid_psnr = psnr(results[f'rgb_{typ}'], rgbs)
        self.valid_psnr.update(valid_psnr)
        self.log('val_psnr', valid_psnr, prog_bar=True, on_step=True, on_epoch=False,
                 logger=True if self.hparams.use_wandb else False)
        return {'val_loss' : self.loss(results, rgbs),
                'val_psnr' : psnr(results[f'rgb_{typ}'], rgbs)}

    def on_validation_epoch_end(self) -> None:
        valid_mean_loss = self.train_loss.compute()
        valid_mean_psnr = self.train_psnr.compute()
        self.train_loss.reset()
        self.train_psnr.reset()
        self.log('val_mean_loss', valid_mean_loss, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.hparams.use_wandb else False)
        self.log('val_mean_psnr', valid_mean_psnr, prog_bar=True, on_step=False, on_epoch=True,
                 logger=True if self.hparams.use_wandb else False)


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(dirpath=f'{hparams.exp_name}/ckpts',
                                          filename='{epoch}-{val_mean_psnr}',
                                          monitor='val_mean_psnr',
                                          mode='min',
                                          save_top_k=3,)

    if hparams.use_wandb:
        if hparams.wandb_id:
            wandb_logger = WandbLogger(
                    project=hparams.project_name,
                    log_model=True,
                    id=hparams.wandb_id,
                    resume="must")
        else:
             wandb_logger = WandbLogger(
                    project=hparams.project_name,
                    log_model=True)
    else:
        wandb_logger=None

    trainer = Trainer(accelerator='auto',
                      max_epochs=hparams.num_epochs,
                      callbacks=checkpoint_callback,
                      logger=wandb_logger,
                      devices=hparams.devices,
                      precision='16',
                      log_every_n_steps=1,
                      # num_sanity_val_steps=1,
                      benchmark=True)
    if hparams.ckpt_path:
        trainer.fit(system, ckpt_path=hparams.ckpt_path)
    else:
        trainer.fit(system)