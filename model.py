"""simple transformer encoder-decoder model
@yashbonde-31.08.2020"""

import torch
import math
import numpy as np

from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss

from prepare_data import VOCAB


class TransformerEncoderDecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # after serious debate I have decided to go with the oob transformer module
        # because of it's inherent simplicity. I tried to play around with the minGPT
        # but it was too tiring, note that is just the enc-dec model, we still need
        # to build embedddings and heads
        self.trans_model = nn.Transformer(
            d_model = config.n_embd,
            nhead = config.n_head,
            num_encoder_layers= config.n_layer,
            num_decoder_layers=config.n_layer,
            dim_feedforward=config.n_embd * 4,
            activation= "gelu",
            dropout= config.pdrop
        )

        # for encoder embedding
        self.linx = nn.Linear(1, config.n_embd)
        self.liny = nn.Linear(1, config.n_embd)
        self.linz = nn.Linear(1, config.n_embd)
        self.lint = nn.Linear(1, config.n_embd)
        self.lino = nn.Linear(1, config.n_embd)
        self.embd = nn.Embedding(5, config.n_embd)

        # for decoder embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.maxlen, config.n_embd))

        # for decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # params
        self.apply(self._init_weights)
        self.maxlen = config.maxlen
        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self,
            x, y, z, t, o, mask_x, mask_y, mask_z, mask_t,
            input_ids, attention_mask,
            targets=None, verbose = False
        ):
        B, Ttgt = input_ids.size()

        if verbose:
            print("---- FORWARD ----")


        if Ttgt > self.maxlen:
            raise ValueError("Current input is longer than maximum allowed length")

        # encoder embedding
        embd_shape = x[:, :, -1].shape
        x = self.linx(x) + self.embd(torch.ones(embd_shape).long() * 0) # [B, N, n_embd]
        y = self.liny(y) + self.embd(torch.ones(embd_shape).long() * 1) # [B, N, n_embd]
        z = self.linz(z) + self.embd(torch.ones(embd_shape).long() * 2) # [B, N, n_embd]
        t = self.lint(t) + self.embd(torch.ones(embd_shape).long() * 3) # [B, N, n_embd]
        o = self.lino(o) + self.embd(torch.ones(embd_shape).long() * 4) # [B, N, n_embd]

        # next we mask the output
        x = x * mask_x.view(-1, 1, 1) # [B, N, n_embd]
        y = x * mask_y.view(-1, 1, 1) # [B, N, n_embd]
        z = x * mask_z.view(-1, 1, 1) # [B, N, n_embd]
        t = x * mask_t.view(-1, 1, 1) # [B, N, n_embd]
        src = x + y + z + t + o # [B, N, n_embd]

        # decoder embedding
        token_embeddings = self.wte(input_ids) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :Ttgt, :] # each position maps to a (learnable) vector
        tgt = token_embeddings + position_embeddings

        if verbose:
            print(f"Source: {src.size()}, Target: {tgt.size()}")

        # run the model and get logits --> we need to transpose because of the 
        dec_out = self.trans_model(
            src = src.transpose(0, 1),
            tgt = tgt.transpose(0, 1),
            tgt_key_padding_mask=attention_mask
        ).transpose(0, 1)
        lm_logits = self.lm_head(self.ln_f(dec_out))

        if verbose:
            print(f"Logits: {lm_logits.size()}")

        # if the targets are given then
        loss = None
        if targets is not None:
            lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            targets = targets.contiguous().view(-1)
            loss_lm_fn = CrossEntropyLoss()
            loss = loss_lm_fn(input = lm_logits, target = targets)
        return lm_logits, loss


# ---- configs ---- #
class Config:
    n_embd = 64
    n_head = 2
    n_layer = 6
    pdrop = 0.1
    vocab_size = len(VOCAB)
    maxlen = 20

    def __init__(self, **kwargs):
        self.attrs = []
        self.pre_attr = ["n_embd", "n_head", "n_layer", "pdrop", "vocab_size", "maxlen",]
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "----- CONFIGURATION -----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in self.attrs + self.pre_attr])

if __name__ == "__main__":
    config = Config(batch_size = 4, num_samples = 14)
    print(config)

    print("---- MODEL ----")
    model = TransformerEncoderDecoderModel(config)
    # print(model)

    def get_dummy_encoder_input():
        return {
            "x": torch.randn(config.batch_size, config.num_samples, 1),
            "y": torch.randn(config.batch_size, config.num_samples, 1),
            "z": torch.randn(config.batch_size, config.num_samples, 1),
            "t": torch.randn(config.batch_size, config.num_samples, 1),
            "o": torch.randn(config.batch_size, config.num_samples, 1),
            "mask_x": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
            "mask_y": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
            "mask_z": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
            "mask_t": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
        }

    def get_dummy_decoder_input():
        attention_mask = []
        for _ in range(config.batch_size):
            attn = [1, ]*(config.maxlen - 4)
            attn += [0, ]*(config.maxlen + 1 - len(attn))
            attention_mask.append(attn)
        return {
            "input_ids": torch.from_numpy(
                np.random.randint(config.vocab_size, size=(config.batch_size, config.maxlen + 1))
            ),
            "attention_mask": torch.from_numpy(np.asarray(attention_mask).astype(np.float32))
        }

    
    print("---- Encoder ----")
    encoder_inputs = get_dummy_encoder_input()
    for k, v in encoder_inputs.items():
        print(f"{k} --> {v.shape}")

    
    print("---- DECODER ----")
    decoder_inputs = get_dummy_decoder_input()
    for k, v in decoder_inputs.items():
        print(f"{k} --> {v.shape}")

    # now we feed shit to the model --> w/o loss
    logits, loss = model(
        **encoder_inputs,
        **{k:v[:,:-1] for k,v in decoder_inputs.items()},
        targets = decoder_inputs["input_ids"][:, 1:],
        verbose = True
    )

    print("Predictions:", torch.argmax(logits,dim  = -1), f"Loss: {loss}")

    # okay cool, if this works till now then we can start training this shit
    optim = model.configure_optimizers()


