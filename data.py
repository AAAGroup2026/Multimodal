"""Load preprocessed MOSI and MOSEI sentiment features."""
import logging
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger("rimoe")


class MMDataset(Dataset):
    def __init__(self, args, mode="train"):
        if args.dataset_name not in ("mosi", "mosei"):
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")
        if mode not in ("train", "valid", "test"):
            raise ValueError(f"Unsupported split: {mode}")
        self.args = args
        self.mode = mode
        with open(args.featurePath, "rb") as stream:
            split = pickle.load(stream)[mode]
        text_key = "text_bert" if args.use_bert else "text"
        self.text = np.asarray(split[text_key], dtype=np.float32)
        self.audio = np.asarray(split["audio"], dtype=np.float32)
        self.vision = np.asarray(split["vision"], dtype=np.float32)
        self.raw_text = split["raw_text"]
        self.ids = split["id"]
        self.labels = np.asarray(split["regression_labels"], dtype=np.float32)
        if not args.need_data_aligned:
            self.audio_lengths = split["audio_lengths"]
            self.vision_lengths = split["vision_lengths"]
        self.audio[self.audio == -np.inf] = 0
        logger.info("%s samples: %s", mode, self.labels.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        sample = {
            "raw_text": self.raw_text[index],
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "index": index,
            "id": self.ids[index],
            "labels": {"M": torch.from_numpy(self.labels[index].reshape(-1))},
        }
        if not self.args.need_data_aligned:
            sample["audio_lengths"] = self.audio_lengths[index]
            sample["vision_lengths"] = self.vision_lengths[index]
        return sample
