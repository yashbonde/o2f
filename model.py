"""simple transformer encoder-decoder model
@yashbonde-31.08.2020"""

import time
import math
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from prepare_data import VOCAB


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, mode = "encoder"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.pdrop)
        self.resid_drop = nn.Dropout(config.pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        if mode == "encoder":
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.encoder_maxlen, config.encoder_maxlen)).view(
                    1, 1, config.encoder_maxlen, config.encoder_maxlen
                ))
            self.is_decoder = False
        elif mode == "decoder":
            self.register_buffer(
                "mask",
                torch.tril(torch.ones(config.decoder_maxlen, config.decoder_maxlen)).view(
                    1, 1, config.decoder_maxlen, config.decoder_maxlen
                ))
            self.is_decoder = True
        else:
            raise ValueError(f"mode can be 'decoder'/'encoder' got: {mode}")

        self.n_head = config.n_head

        self.output_attentions = False
        if hasattr(config, "output_attentions"):
            self.output_attentions = config.output_attentions

    def forward(self, tgt, memory = None, forward_mask = False, pad_mask = None):
        # this is going to be the case with encoders and decoder bottom layer
        if memory == None:
            memory = tgt
        Btgt, target_seqlen, C = tgt.size()
        Bmem, memory_seqlen, _ = memory.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q = self.query(tgt).view(Btgt, target_seqlen, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = self.key(memory).view(Bmem, memory_seqlen, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(memory).view(Bmem, memory_seqlen, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T_tgt, hs) x (B, nh, hs, T_mem) -> (B, nh, T_tgt, T_mem)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if forward_mask:
            att = att.masked_fill(self.mask[:,:,:target_seqlen,:target_seqlen] == 0, float('-inf'))

        if pad_mask is not None:
            # pad_mask: [B, target_seqlen] --> [B, target_seqlen, target_seqlen]
            att = att + pad_mask.unsqueeze(1)

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(Btgt, target_seqlen, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        out = (y,)
        if self.output_attentions:
            out = (y, att)
        return out


class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, mode = "encoder")
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.pdrop),
        )
        self.openai_block = config.openai_block

        self.output_attentions = False
        if hasattr(config, "output_attentions"):
            self.output_attentions = config.output_attentions

    def forward(self, d):
        # in OpenAI implementation they perform normalisation before attention and MLP
        x = d[0]

        if self.openai_block: x = self.ln1(x)
        self_att = self.attn(x)
        x = x + self_att[0]
        if not self.openai_block: x = self.ln1(x)
        
        # MLP
        if self.openai_block: x = self.ln2(x)
        x = x + self.mlp(x)
        if not self.openai_block: x = self.ln2(x)

        o = (x,)
        if self.output_attentions:
            oattn = ((self_att[1],),) + d[1]
            o = (x, oattn)
        return o


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, mode="decoder")
        self.attn = CausalSelfAttention(config, mode="decoder")
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.pdrop),
        )

        self.openai_block = config.openai_block

        self.output_attentions = False
        if hasattr(config, "output_attentions"):
            self.output_attentions = config.output_attentions


    def forward(self, d):
        # in OpenAI implementation they perform normalisation before attention and MLP
        x, mem, pad_mask = d[:3]

        # self-attention
        if self.openai_block: x = self.ln1(x)
        self_att = self.attn(x, forward_mask=True, pad_mask=pad_mask)
        x = x + self_att[0]
        if not self.openai_block: x = self.ln1(x)

        # memory-attention
        if self.openai_block: x = self.ln2(x)
        mem_att = self.attn(x, mem)
        x = x + mem_att[0]
        if not self.openai_block: x = self.ln2(x)

        # MLP
        if self.openai_block: x = self.ln3(x)
        x = x + self.mlp(x)
        if not self.openai_block: x = self.ln3(x)

        o = (x, mem, pad_mask,)
        if self.output_attentions:
            oattn = ((self_att[1], mem_att[1],), *d[3])
            o = (x, mem, pad_mask, oattn)
        return o


class TransformerEncoderDecoderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # after serious debate I have decided to go with the oob transformer module
        # because of it's inherent simplicity. I tried to play around with the minGPT
        # but it was too tiring, note that is just the enc-dec model, we still need
        # to build embedddings and heads
        # self.trans_model = nn.Transformer(
        #     d_model = config.n_embd,
        #     nhead = config.n_head,
        #     num_encoder_layers= config.n_layer,
        #     num_decoder_layers=config.n_layer,
        #     dim_feedforward=config.n_embd * 4,
        #     activation= "gelu",
        #     dropout= config.pdrop
        # )

        # oh yeah, I got it working!

        self.encoder = nn.Sequential(*[EncoderBlock(config) for _ in range(config.n_layer)])
        self.decoder = nn.Sequential(*[DecoderBlock(config) for _ in range(config.n_layer)])

        # for encoder embedding
        self.linx = nn.Sequential(nn.Linear(1, config.n_embd), nn.LayerNorm(config.n_embd))
        self.liny = nn.Sequential(nn.Linear(1, config.n_embd), nn.LayerNorm(config.n_embd))
        self.linz = nn.Sequential(nn.Linear(1, config.n_embd), nn.LayerNorm(config.n_embd))
        self.lint = nn.Sequential(nn.Linear(1, config.n_embd), nn.LayerNorm(config.n_embd))
        self.lino = nn.Sequential(nn.Linear(1, config.n_embd), nn.LayerNorm(config.n_embd))
        self.embd = nn.Embedding(5, config.n_embd)

        # for decoder embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.decoder_maxlen, config.n_embd))

        # for decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # params
        self.apply(self._init_weights)
        self.decoder_maxlen = config.decoder_maxlen
        self.n_embd = config.n_embd
        self.use_var_masking = config.use_var_masking
        self.num_params = sum(p.numel() for p in self.parameters())
        logging.info(f"number of parameters: {self.num_params}")

        self.openai_block = config.openai_block
        self.use_emb_matrix_head = config.use_emb_matrix_head

        self.output_attentions = False
        if hasattr(config, "output_attentions"):
            self.output_attentions = config.output_attentions

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
            targets=None, verbose = False,
        ):
        if verbose:
            print("---- FORWARD ----")

        # embed the input and pass through the encoder layers
        enc_out = self.enc_out(
            x, y, z, t, o,
            mask_x, mask_y, mask_z, mask_t,
            verbose,
        )

        # take the embeddings from encoder and input to decoder
        # and get the output logits
        lm_logits, dec_attn = self.dec_out(
            enc_out[0], input_ids, attention_mask, verbose
        )
        if verbose:
            print(f"Logits: {lm_logits.size()}")

        # if the targets are given then
        loss = None
        if targets is not None:
            lm_logits = lm_logits.view(-1, lm_logits.size(-1))
            targets = targets.contiguous().view(-1)
            att_loss = attention_mask.contiguous().view(-1) # flatten for loss

            # we flatten the loss and apply attention mask because we do not need
            # to train model on [PAD] tokens
            loss = F.cross_entropy(lm_logits, targets, reduce=False)
            loss = loss * att_loss
            loss = torch.mean(loss)

        o = (lm_logits, loss)
        if self.output_attentions:
            enc_attn = enc_out[-1]
            dec_attn = dec_attn[-1]
            o = (lm_logits, loss, (enc_attn, dec_attn))
        return o


    def enc_out(self,
        x, y, z, t, o, mask_x, mask_y, mask_z, mask_t,
        verbose = False
    ):
        """method for a single encoder forward pass, useful when beam decoding"""
        # encoder embedding
        embd_shape = x.size()[:2]
        x = self.linx(x) + self.embd(torch.ones(embd_shape).long() * 0) # [B, N, n_embd]
        y = self.liny(y) + self.embd(torch.ones(embd_shape).long() * 1) # [B, N, n_embd]
        z = self.linz(z) + self.embd(torch.ones(embd_shape).long() * 2) # [B, N, n_embd]
        t = self.lint(t) + self.embd(torch.ones(embd_shape).long() * 3) # [B, N, n_embd]
        o = self.lino(o) + self.embd(torch.ones(embd_shape).long() * 4) # [B, N, n_embd]

        # mask encoder input based on wether the variables are present or not
        if self.use_var_masking:
            x = x * mask_x.view(-1, 1, 1) # [B, N, n_embd]
            y = x * mask_y.view(-1, 1, 1) # [B, N, n_embd]
            z = x * mask_z.view(-1, 1, 1) # [B, N, n_embd]
            t = x * mask_t.view(-1, 1, 1) # [B, N, n_embd]
        src = x + y + z + t + o # [B, N, n_embd]
        if verbose: print(f"Source: {src.size()}")

        if self.output_attentions:
            src = (src, ())
        enc_out = self.encoder(src)
        return enc_out

    def dec_out(self, enc_out, input_ids, attention_mask, verbose = False):
        B, Ttgt = input_ids.size()

        if Ttgt > self.decoder_maxlen:
            raise ValueError(f"Current input is longer than maximum allowed length, got: {Ttgt}")

        # decoder embedding
        token_embeddings = self.wte(input_ids) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :Ttgt, :] # each position maps to a (learnable) vector
        tgt = token_embeddings + position_embeddings

        # get the padding mask
        B, trgsz = attention_mask.size()
        lens = torch.argmax(attention_mask, dim=1)  # [B,]
        pad_mask = torch.ones((B, trgsz, trgsz), requires_grad=False)
        for b, l in zip(range(B), lens):
            pad_mask[b, :l, :l] = 0
        pad_mask[pad_mask == 1.] = -1e10

        if verbose:
            print(f"Target: {tgt.size()}")
            print(f"pad_mask: {pad_mask.size()}")
            print(f"attention_mask: {attention_mask.size()}")

        # decoder output and langauge model head
        dec_in = (tgt, enc_out, pad_mask)
        if self.output_attentions:
            dec_in = (tgt, enc_out, pad_mask, ())
        out = self.decoder(dec_in)
        dec_out = self.ln_f(out[0])
        if self.use_emb_matrix_head:
            lm_logits = F.linear(dec_out, self.wte.weight)
        else:
            lm_logits = self.lm_head(dec_out)

        del pad_mask # conserve memory
        return lm_logits, out


# ---- trainer ---- #
class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config, optimizer = None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logging.info(f"Saving Model at {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self, verbose = False):
        model, config = self.model, self.config
        optimizer = self.optimizer if self.optimizer is not None else model.configure_optimizers(config)
        gs = 0
        lr = config.learning_rate

        tbtrue = True if hasattr(config, "tb_path") else False
        if tbtrue:
            tb = SummaryWriter(log_dir=config.tb_path, flush_secs=20)

        def run_epoch(split, gs = None, lr = None):
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            dl = DataLoader(
                data,
                shuffle = True,
                pin_memory = True,
                batch_size = config.batch_size,
                num_workers = config.num_workers
            )

            losses = []
            pbar = tqdm(enumerate(dl))
            for it, (enc, dec) in pbar:
                _l = -1 if not losses else losses[-1]
                _lr = -1 if lr is None else lr
                
                if is_train:
                    pbar.set_description(f"[TRAIN] GS: {gs}, LR: {round(_lr, 5)}, Loss: {round(_l, 5)}")
                else:
                    pbar.set_description(f"[VAL]")
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(
                        **enc,
                        **{k: v[:, :-1] for k, v in dec.items()},
                        targets=dec["input_ids"][:, 1:],
                        verbose=verbose
                    )

                    if str(loss.item()) == "nan":
                        print(enc)
                        print(dec)
                        exit()

                    losses.append(loss.item())

                if is_train:
                    if tbtrue and gs:
                        # add things to tb
                        tb.add_scalar("loss", loss.item(), global_step=gs, walltime=time.time())
                        tb.add_scalar("lr", lr, global_step=gs, walltime=time.time())

                    # back prob and update the gradient
                    for p in model.parameters(): # better than model.zero_grad()
                        p.grad = None
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()
                    gs += 1

                    # use the noam scheme from original transformers papers
                    lr = min(gs**(-0.5), gs/(config.warmup_steps ** 1.5)) * (model.n_embd ** -0.5) * 0.1
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
            
            if not is_train:
                test_loss = float(np.mean(losses))
                return test_loss

            return gs, lr

        # now write wrapper for each epoch
        best_loss = float("inf")
        for epoch in range(config.max_epochs):
            gs, lr = run_epoch("train", gs, lr)
            if self.test_dataset is not None:
                test_loss = run_epoch("test")
                logging.info(f"Test loss: {test_loss}")
            
            # early stopping based on the test loss of just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()
        
        # close if used
        if tbtrue:
            tb.close()


# ---- configs ---- #
class Config:
    n_embd = 64
    n_head = 2
    n_layer = 6
    pdrop = 0.1
    vocab_size = len(VOCAB)
    encoder_maxlen = 40
    decoder_maxlen = 20
    use_var_masking = True

    output_attentions = False
    openai_block = True
    use_emb_matrix_head = False

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- MODEL CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "n_embd",
                "n_head",
                "n_layer",
                "pdrop",
                "vocab_size",
                "encoder_maxlen",
                "decoder_maxlen",
                "output_attentions",
                "openai_block",
                "use_emb_matrix_head"
            ] + self.attrs)) 
        ]) + "\n"


class TrainerConfig:
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights

    lr_decay = False
    warmup_steps = 1000

    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        self.attrs = []
        for k,v in kwargs.items():
            setattr(self, k, v)
            self.attrs.append(k)

    def __repr__(self):
        return "---- TRAINER CONFIGURATION ----\n" + \
            "\n".join([f"{k}\t{getattr(self, k)}" for k in list(set([
                "max_epochs",
                "batch_size",
                "learning_rate",
                "betas",
                "grad_norm_clip",
                "weight_decay",
                "ckpt_path",
                "warmup_steps",
                "num_workers",
            ] + self.attrs))
        ]) + "\n"


if __name__ == "__main__":
    config = Config(batch_size=4, encoder_maxlen=14, use_emb_matrix_head=True)
    print(config)

    print("---- MODEL ----")
    model = TransformerEncoderDecoderModel(config)
    # print(model)

    def get_dummy_encoder_input():
        return {
            "x": torch.randn(config.batch_size, config.encoder_maxlen, 1),
            "y": torch.randn(config.batch_size, config.encoder_maxlen, 1),
            "z": torch.randn(config.batch_size, config.encoder_maxlen, 1),
            "t": torch.randn(config.batch_size, config.encoder_maxlen, 1),
            "o": torch.randn(config.batch_size, config.encoder_maxlen, 1),
            "mask_x": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
            "mask_y": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
            "mask_z": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
            "mask_t": torch.from_numpy(np.random.randint(2, size = (config.batch_size))),
        }

    def get_dummy_decoder_input():
        attention_mask = []
        for _ in range(config.batch_size):
            attn = [1, ]*(config.decoder_maxlen - 4)
            attn += [0, ]*(config.decoder_maxlen + 1 - len(attn))
            attention_mask.append(attn)
        return {
            "input_ids": torch.from_numpy(
                np.random.randint(config.vocab_size, size=(
                    config.batch_size, config.decoder_maxlen + 1))
            ),
            "attention_mask": torch.from_numpy(np.asarray(attention_mask).astype(np.float32))
        }
    
    print("---- Encoder ----")
    encoder_inputs = get_dummy_encoder_input()
    for k, v in encoder_inputs.items():
        print(f"{k} --> {(v.shape, v.dtype)}")

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

#     print("Predictions:", torch.argmax(logits,dim  = -1), f"Loss: {loss}")

#     # okay cool, if this works till now then we can start training this shit
#     from types import SimpleNamespace
#     from prepare_data import O2fDataset

#     data_config = SimpleNamespace(
#         Nmin=1,
#         Nmax=4,
#         p1min=1,
#         p1max=3,
#         p2min=1,
#         p2max=3,
#         lmin=1,
#         lmax=3,
#         num_samples=40,
#         dataset_size=10,
#         maxlen=20
#     )
#     train_conf = TrainerConfig(max_epochs=1)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=train_conf.learning_rate, betas=train_conf.betas)

#     ds = O2fDataset(data_config)
#     DataLoader(ds, batch_size=train_conf.batch_size, shuffle=True)
#     trainer = Trainer(model, optimizer, ds, None, train_conf)
#     trainer.train()

# ------ THERE WAS A NAN ISSUE ------ #
# if __name__ == "__main__":
#     config = Config(
#         n_embd=32,
#         n_head=1,
#         n_layer=1,
#         pdrop=0.1,
#         vocab_size=39,
#         maxlen=20
#     )
#     model = TransformerEncoderDecoderModel(config)

#     from _nan import * # temp

#     # checking why am I getting nans from forward pass
#     def nan_hook(self, inp, output):
#         if not isinstance(output, tuple):
#             outputs = [output]
#         else:
#             outputs = output

#         for i, out in enumerate(outputs):
#             nan_mask = torch.isnan(out)
#             if nan_mask.any():
#                 print("In", self.__class__.__name__)
#                 raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(
#                 ), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

#     # for submodule in model.modules():
#     #     submodule.register_forward_hook(nan_hook)

#     for k, v in enc.items():
#         enc[k] = torch.from_numpy(np.asarray(v).astype(np.float32))
#     for k, v in dec.items():
#         dec[k] = torch.from_numpy(np.asarray(v)).long()

#     out = model(**{k:v[1:] for k,v in enc.items()}, **{k:v[1:,:-1] for k,v in dec.items()}, verbose = True)
#     print(out)
