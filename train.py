"""main o2f trainer script
02.10.2020 - @yashbonde"""

import os
import json
import logging
import numpy as np
from argparse import ArgumentParser

import torch
from torch._C import dtype
from torch.utils.data import Dataset, DataLoader

from model import Config, TrainerConfig, Trainer
from model import TransformerEncoderDecoderModel
from utils import show_notification

logging.basicConfig(level = logging.INFO)

args = ArgumentParser(description="script to train o2f model")

# ---- args ---- #
args.add_argument("--model_folder", default="models", type=str,
                  help="folder where all models are")
args.add_argument("--name", default=None, type=str,
                  help="folder for this model")
args.add_argument("--data_json", default=None, type=str,
                  help="path to data json")
args.add_argument("--train_split", default=0.9, type=float,
                  help="train size split")

# ---- config ---- #
args.add_argument("--n_embd", default=128, type=int,
                  help="embedding dimension of the model")
args.add_argument("--n_head", default=8, type=int,
                  help="number of heads for multihead attention")
args.add_argument("--n_layer", default=6, type=int,
                  help="number of stacks in encoder and decoder")
args.add_argument("--encoder_maxlen", default=100, type=int,
                  help="maximum length of encoder input")
args.add_argument("--decoder_maxlen", default=40, type=int,
                  help="maximum length of decoder input")
args.add_argument("--use_var_masking", default=False, type=bool,
                  help="apply mask for variables")
args.add_argument("--pdrop", default=0.1, type=float,
                  help="dropout probability")

# ---- trainer ---- #
args.add_argument("--epochs", default=20, type=int,
                  help="number of epochs to train the model")
args.add_argument("--batch_size", default=128, type=int,
                  help="batch size for training")
args.add_argument("--learning_rate", default=3e-4,
                  type=float, help="initial learning rate")
args.add_argument("--beta1", default=0.9, type=float,
                  help="beta_1 parameter for AdamW")
args.add_argument("--beta2", default=0.95, type=float,
                  help="beta_2 parameter for AdamW")
args.add_argument("--grad_norm_clip", default=1.0,
                  type=float, help="gradient clipping value")
args.add_argument("--warmup_steps", default=60,
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
    pdrop=args.pdrop,
    encoder_maxlen = args.encoder_maxlen,
    decoder_maxlen=args.decoder_maxlen,
    use_var_masking=args.use_var_masking
)

logging.info(trainer_conf)
logging.info(config)

# make dirs
os.makedirs(args.model_folder, exist_ok=True)
os.makedirs(trainer_conf.tb_path, exist_ok=False)

# load a dataset
class Ds(Dataset):
    def __init__(self, mode: str):
        if mode not in ["test", "train"]:
            raise ValueError("Dataset mode onl")
        with open(args.data_json, "r") as f:
            logging.info(f"Loading file for: `{mode}` mode")
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
optimizer = None
# optimizer = torch.optim.AdamW(
#     model.parameters(),
#     lr=trainer_conf.learning_rate,
#     betas=trainer_conf.betas
# )
# DataLoader(, batch_size=trainer_conf.batch_size, shuffle=True)
ds_train = Ds("train")
ds_test = Ds("test")

# logging.info("======= TRAIN =======")
# for i, (_enc, _dec) in enumerate(DataLoader(ds_train, batch_size=32, shuffle = True)):
#     for k, v in _enc.items():
#         logging.info(f"{k} ({v.size()},{v.dtype})")
#     for k, v in _dec.items():
#         logging.info(f"{k} ({v.size()},{v.dtype})")
#     break

# logging.info("======= TEST =======")
# for i, (_enc, _dec) in enumerate(DataLoader(ds_test, batch_size=32, shuffle=True)):
#     for k, v in _enc.items():
#         logging.info(f"{k} ({v.size()},{v.dtype})")
#     for k, v in _dec.items():
#         logging.info(f"{k} ({v.size()},{v.dtype})")
#     break

trainer = Trainer(model, ds_train, ds_test, trainer_conf, optimizer=optimizer)
try:
    trainer.train()
except Exception as e:
    logging.error(f"Exception: {e} has occured", exc_info = True)
    show_notification("o2f Training", f"Exception {e} has occured")

show_notification("o2f Training", "Training Complete")
