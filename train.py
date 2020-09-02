"""main o2f trainer script
02.10.2020 - @yashbonde"""

import os
from argparse import ArgumentParser
import json

from model import TrainerConfig, Trainer, TransformerEncoderDecoderModel, Config

args = ArgumentParser(description="script to train o2f model")

# ---- saving ---- #
model_folder = None
name = None

# ---- config ---- #
n_embd = 64
n_head = 2
n_layer = 6
pdrop = 0.1

# ---- trainer ---- #
max_epochs = 10
batch_size = 64
learning_rate = 3e-4
betas = (0.9, 0.95)
grad_norm_clip = 1.0
warmup_steps = 1000

args = args.parse_args()
args = (vars(args))

trainer_conf = TrainerConfig(
    max_epochs= args.max_epochs,
    batch_size= args.batch_size,
    learning_rate= args.learning_rate,
    betas= (args.beta1, args.beta2),
    grad_norm_clip= args.grad_norm_clip,
    warmup_steps= args.warmup_steps,
    ckpt_path=os.path.join(args.model_folder, args.name)
)
config = Config(n_embd=args.n_embd,
                n_head=args.n_head,
                n_layer=args.n_layer,
                pdrop=args.pdrop)
