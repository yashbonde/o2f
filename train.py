"""main o2f trainer script
02.10.2020 - @yashbonde"""

import os
import json
import numpy as np
from argparse import ArgumentParser

import torch
from torch.utils.data import Dataset, DataLoader

from model import Config, TrainerConfig, TransformerEncoderDecoderModel, Trainer

args = ArgumentParser(description="script to train o2f model")

# ---- args ---- #
args.add_argument("--model_folder", default=None, type=str,
                  help="folder where all models are")
args.add_argument("--name", default=None, type=str,
                  help="folder for this model")
args.add_argument("--data_json", default=None, type=str,
                  help="path to data json")
args.add_argument("--train_split", default=0.9, type=float,
                  help="train size split")

# ---- config ---- #
args.add_argument("--n_embd", default=64, type=int,
                  help="embedding dimension of the model")
args.add_argument("--n_head", default=2, type=int,
                  help="number of heads for multihead attention")
args.add_argument("--n_layer", default=6, type=int,
                  help="number of stacks in encoder and decoder")
args.add_argument("--pdrop", default=0.1, type=float,
                  help="dropout probability")

# ---- trainer ---- #
args.add_argument("--epochs", default=10, type=int,
                  help="number of epochs to train the model")
args.add_argument("--batch_size", default=64, type=int,
                  help="batch size for training")
args.add_argument("--learning_rate", default=3e-4,
                  type=float, help="initial learning rate")
args.add_argument("--beta1", default=0.9, type=float,
                  help="beta_1 parameter for AdamW")
args.add_argument("--beta2", default=0.95, type=float,
                  help="beta_2 parameter for AdamW")
args.add_argument("--grad_norm_clip", default=1.0,
                  type=float, help="gradient clipping value")
args.add_argument("--warmup_steps", default=1000,
                  type=int, help="warmup steps")
args = args.parse_args()

# define configurations
trainer_conf = TrainerConfig(
    max_epochs=args.epochs,
    batch_size=args.batch_size,
    learning_rate=args.learning_rate,
    betas=(args.beta1, args.beta2),
    grad_norm_clip=args.grad_norm_clip,
    warmup_steps=args.warmup_steps,
    ckpt_path=os.path.join(args.model_folder, args.name, f"{args.name}.pt"),
    tb_path=os.path.join(args.model_folder, args.name),
)
config = Config(
    n_embd=args.n_embd,
    n_head=args.n_head,
    n_layer=args.n_layer,
    pdrop=args.pdrop
)

print(trainer_conf)
print(config)

# make dirs
os.makedirs(args.model_folder, exist_ok=True)
os.makedirs(trainer_conf.tb_path, exist_ok=True)

# load a dataset
class Ds(Dataset):
    def __init__(self, mode: str):
        if mode not in ["test", "train"]:
            raise ValueError("Dataset mode onl")
        with open(args.data_json, "r") as f:
            data = json.load(f)

        # convert to line wise indexing
        split_idx = int(args.train_split * len(data["encoder"]["x"]))
        self.data = []
        if mode == "train": # till split_idx
            r = range(0, split_idx)
        else: # from split_idx
            r = range(split_idx, len(data["encoder"]["x"]))
        for i in r:
            self.data.append(({
                "x": data["encoder"]["x"][i],
                "y": data["encoder"]["y"][i],
                "z": data["encoder"]["z"][i],
                "t": data["encoder"]["t"][i],
                "o": data["encoder"]["o"][i],
                "mask_x": data["encoder"]["mask_x"][i],
                "mask_y": data["encoder"]["mask_y"][i],
                "mask_z": data["encoder"]["mask_z"][i],
                "mask_t": data["encoder"]["mask_t"][i],
            }, {
                "input_ids": data["decoder"]["input_ids"][i],
                "attention_mask": data["decoder"]["attention_mask"][i],
            }))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        enc, dec = self.data[idx]
        for k, v in dec.items():
            dec[k] = torch.from_numpy(np.asarray(v)).long()
        for k, v in enc.items():
            if "mask" in k:
                enc[k] = torch.squeeze(torch.from_numpy(np.asarray(v)), -1)
            else:
                # print(v)
                v = np.asarray(v).astype(np.float32)
                enc[k] = torch.from_numpy(v)
        return enc, dec

# create models and dataloader
model = TransformerEncoderDecoderModel(config)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=trainer_conf.learning_rate,
    betas=trainer_conf.betas
)
# DataLoader(, batch_size=trainer_conf.batch_size, shuffle=True)
ds_train = Ds("train")
ds_test = Ds("test")

# print("======= TRAIN =======")
# for i, (_enc, _dec) in enumerate(DataLoader(ds_train, batch_size=32)):
#     print("---> enc:", {k: (v.size(), v.dtype) for k, v in _enc.items()})
#     print("---> dec:", {k: (v.size(), v.dtype) for k, v in _dec.items()})
#     break

# print("======= TEST =======")
# for i, (_enc, _dec) in enumerate(DataLoader(ds_test, batch_size=32)):
#     print("---> enc:", {k: (v.size(), v.dtype) for k, v in _enc.items()})
#     print("---> dec:", {k: (v.size(), v.dtype) for k, v in _dec.items()})
#     break

trainer = Trainer(model, optimizer, ds_train, ds_test, trainer_conf)
trainer.train()



