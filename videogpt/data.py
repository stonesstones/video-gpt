import numpy as np

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
import pytorch_lightning as pl
import json
from PIL import Image


class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.data_folder = data_folder
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        train_json_path = "./videos_train_path.json"
        val_json_path = "./videos_val_path.json"
        print(f"make train={train} dataset")
        if train:
            with open(train_json_path, 'r') as f:
                self.data = json.load(f)
        else:
            with open(val_json_path, 'r') as f:
                self.data = json.load(f)
        self.episodes = self.data['episodes']

        print(f"self.episodes: {len(self.episodes)}")
        self.sequences = []
        # nusc fps default 12.5 -> 4
        # for episode in self.episodes:
        #     episode["paths"] = episode["paths"][::3]
        for episode_idx, episode in enumerate(self.episodes):
            num_frames = len(episode['paths'])
            if num_frames >= sequence_length:
                num_sequences = num_frames - sequence_length + 1
                for seq_idx in range(0, num_sequences, sequence_length // 2): # sequence_length // 2 is the stride
                    self.sequences.append({
                        'episode_idx': episode_idx,
                        'start_frame': seq_idx,
                        'end_frame': seq_idx + sequence_length
                    })
        print(f"self.sequences: {len(self.sequences)}")
        print(f"self.sequences[0]: {self.sequences[0]}")

    @property
    def n_classes(self):
        return 0

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        data = self.sequences[idx]
        episode_idx = data['episode_idx']
        start_frame = data['start_frame']
        end_frame = data['end_frame']
        episode = self.episodes[episode_idx]
        paths = episode['paths']
        video = [Image.open(path).resize((self.resolution, self.resolution)) for path in paths[start_frame:end_frame]]
        video = np.array(video)
        video = video.transpose(3, 0, 1, 2) # CTHW
        video = video.astype(np.float32) / 255.0
        video -= 0.5
        video = torch.from_numpy(video)
        return dict(video=video, label=0)


class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes


    def _dataset(self, train):
        Dataset = VideoDataset
        dataset = Dataset(self.args.data_path, self.args.sequence_length,
                          train=train, resolution=self.args.resolution)
        return dataset


    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()
