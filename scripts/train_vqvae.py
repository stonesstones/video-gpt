import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from videogpt import VQVAE, VideoData
from pytorch_lightning.loggers import WandbLogger


def main():
    print("start training")
    pl.seed_everything(1234)

    parser = argparse.ArgumentParser()

    # VQVAE モデルの引数を追加
    parser = VQVAE.add_model_specific_args(parser)

    # データと訓練用の追加引数
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--data_path', type=str, default='/groups/gcg51472/nuscenes')
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)

    # Trainer 引数を明示的に追加（from_argparse_args を使わないため）
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--sync_batchnorm', action='store_true', help='Use synchronized batch norm')
    parser.add_argument('--max_steps', type=int, default=200000)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--precision', type=int, default=16)
    parser.add_argument('--log_every_n_steps', type=int, default=100)
    parser.add_argument('--limit_val_batches', type=int, default=0.2)
    parser.add_argument('--val_check_interval', type=float, default=1)

    args = parser.parse_args()

    # データセット準備
    data = VideoData(args)
    data.train_dataloader()  # キャッシュ作成などのために一度呼び出す
    data.test_dataloader()

    # モデル初期化
    model = VQVAE(args)

    # コールバックの設定
    callbacks = [
        ModelCheckpoint(monitor='val/recon_loss', mode='min', dirpath=f'./logs/{args.name}'), 
        ModelCheckpoint(every_n_epochs=2, dirpath=f'./logs/{args.name}', filename='{epoch}')
        ]

    wandb_logger = WandbLogger(project="video-gpt", name=args.name)
    # Trainer 構築
    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus,
        strategy='ddp' if args.gpus > 1 else 'auto',
        sync_batchnorm=args.sync_batchnorm if args.gpus > 1 else False,
        max_steps=args.max_steps,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=args.val_check_interval,
    )
    trainer.fit(model, data)
    
if __name__ == '__main__':
    main()
