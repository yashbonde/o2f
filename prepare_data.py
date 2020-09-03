"""~script to generate data for o2f model~
dataloader object for o2f model, saving the data doesn't really make sense
because of large dataset
29.08.2020 - @yashbonde

This is inspired from FAIR paper: Deep Learning for Symbolic Mathematics
http://arxiv.org/abs/1912.01412

For more details read: generate_expressions.md
"""

import re
import time
import errno
import logging
import random
import signal
from functools import wraps, partial

import sympy
from sympy import simplify, Float, Integer, preorder_traversal
import os
from maths import Math
import numpy as np
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader

from utils import show_notification, timeout, set_seed

logger = logging.getLogger('prepare_data')
logger.setLevel(logging.INFO)

# ---- constants ---- #
OPS = {
    "mult": (Math.mult, 2),
    "div": (Math.div, 2),
    "add": (Math.add, 2),
    "sub": (Math.sub, 2),
    "exp": (Math.exp, 1),
    "log": (Math.log, 1),
    "sin": (Math.sin, 1),
    "cos": (Math.cos, 1),
    "tan": (Math.tan, 1),
    "pow": (Math.pow, 2),
    "sqrt": (Math.sqrt, 1)
}  # tuple of <op, arity>

VARIABLES = [i for i in "xyzt"]
COEFFICIENTS = [f"a{i}" for i in range(10)]
CONSTANTS = ["pi", "E"]

ROUND = 3
COEFF = [0, +10.]

UNA_OPS = [k for k, v in OPS.items() if v[1] == 1]
BIN_OPS = [k for k, v in OPS.items() if v[1] == 2]

START_TOKEN = "[start]"
PAD_TOKEN = "[pad]"
END_TOKEN = "[end]"

VOCAB_TOKENS =  [f"_{x}_" for x in list(OPS.keys())] # operations
VOCAB_TOKENS += [f"{i}" for i in range(10)] # numbers
VOCAB_TOKENS += VARIABLES # variables
VOCAB_TOKENS += ["_pi_", "_e_"] # special numbers
VOCAB_TOKENS += ["(", "+", "-", "*", "/", ")", ".", "_**_"] # other math ops
VOCAB_TOKENS += [START_TOKEN, PAD_TOKEN, END_TOKEN]  # language modelling
VOCAB_TOKENS += ["o"] # ??
VOCAB = {t: i for i, t in enumerate(VOCAB_TOKENS)}

# ---- functions ---- #


def get_ubi_dist(N=3, p1=1, p2=1, l=1):
    """goto: generating_expressions.md

    First step is to define all the unary-binary trees it is done in two steps:
    1. iterate over the 2N possible nodes so each index corresponds to a n value
    2. change the axis so each index now correspond to e value

    N = 3 gives the following output:
    (1) --> [[0, 1, 1, 1, 1],
             [0, 2, 4, 6],
             [0, 6, 16],
             [0, 22],
             [0]]
    (2) --> [[0, 0, 0, 0, 0],
             [1, 2, 6, 22],
             [1, 4, 16],
             [1, 6],
             [1]]

    Or think it like this:
        n = 0  1  2  3  4
          +---------------
    e = 0 | 0  0  0  0  0 
    e = 1 | 1  2  6  22
    e = 2 | 1  4  16
    e = 3 | 1  6
    e = 4 | 1

    So you can see that we get D(0,n) = 0, D(e,0) = 1 and so on

    :param N: maximum number of operations to add
    :param p1: number of unary operations
    :param p2: number of binary operations
    :param l: number of leaves in the stack
    """

    # (1)
    D_ubi = []
    D_ubi.append([0] + [l**i for i in range(1, 2*N + 1)])
    for _n in range(1, 2*N + 1):  # number of operators
        s = [0]
        for _e in range(1, 2*N - _n + 1):  # number of empty nodes
            s.append(l * s[_e-1] + p1*D_ubi[_n-1][_e] + p2*D_ubi[_n-1][_e+1])
        D_ubi.append(s)
    assert all(len(D_ubi[i]) >= len(D_ubi[i+1]) for i in range(len(D_ubi) - 1))

    # (2)
    dubi = []
    longest_tree_size = max(len(x) for x in D_ubi)
    for i in range(longest_tree_size):  # iterate over longest tree
        this_ubi = []  # for this iteration
        for _tree in D_ubi:
            if i < len(_tree):
                this_ubi.append(_tree[i])
        dubi.append(this_ubi)
    return dubi


def generate_random_expression(
        dubi,
        N=3,
        l=1,
        p1=1,
        p2=1,
        leaf_probs=(0.55, 0.1, 0.25, 0.1),
        una_probs=None,
        bi_probs=None
    ):
    """goto: generating_expressions.md

    method to generate expressions, inspired from the main source code:
    https://github.com/facebookresearch/SymbolicMathematics/blob/master/src/envs/char_sp.py

    :param dubi: unary-binary distribution
    :param N: maximum number of operators
    :param l: number of leaves in the model
    :param p1: number of unary operators
    :param p2: number of binary operators
    :param leaf_probs: Leaf probabilities of being a variable, a coefficient,
        an integer, or a constant.
    :param una_probs: list of probabilities for unary operators
    :param bi_probs: list of probabilities for binary operators
    """

    # init things
    e = 1  # number of empty leaves
    l_leaves = 0  # left leaves - None state reserved for leaves
    t_leaves = 1  # total number of leaves (for sanity)
    stack = [None]

    # start making tree
    for _n in range(N, 0, -1):
        # L523 -- get the next operator, arity and position

        # L474 -- sample a position of the next node (unary-binary case)
        # sample a position in {0, ..., e-1}, along with arity
        probs = []
        for k in range(e):
            probs.append((l**k) * p1 * dubi[e - k][_n - 1])
        for k in range(e):
            probs.append((l**k) * p2 * dubi[e - k + 1][_n - 1])
        probs = [p / dubi[e][_n] for p in probs]
        probs = np.array(probs, dtype=np.float64)
        k = np.random.choice(2 * e, p=probs)
        a = 1 if k < e else 2  # arity
        k = k % e  # position
        # L491

        # L524 -- now we have sampled the position and arity we will
        # sample the operator
        if a == 1:
            op = np.random.choice(UNA_OPS, p=una_probs)
        else:
            op = np.random.choice(BIN_OPS, p=bi_probs)

        # created empty nodes - skipped future leaves
        e = e + OPS[op][1] - 1 - k
        t_leaves = t_leaves + OPS[op][1] - 1  # update total no. of leaves
        l_leaves = l_leaves + k  # update no. of left leaves

        # update the tree
        pos = [i for i, v in enumerate(stack) if v is None][l_leaves]
        stack = stack[:pos] + [op] + [None for _ in range(OPS[op][1])] + stack[pos + 1:]

    # L545 -- create leaves
    leaves = []
    for _ in range(t_leaves):
        # L498
        leaf_type = np.random.choice(4, p=leaf_probs)
        if leaf_type == 0:  # variable
            leaves.append(list(VARIABLES)[np.random.randint(len(VARIABLES))])
        elif leaf_type == 1:  # coefficient
            leaves.append(list(COEFFICIENTS)[np.random.randint(len(COEFFICIENTS))])
        elif leaf_type == 2:  # integer
            c = np.random.randint(low=COEFF[0], high=COEFF[1])
            c = c if np.random.randint(2) == 0 else -c
            out = [x for x in str(c)]
            if out[0] == "-":
                out[0] = "INT-"
            else:
                out = ["INT"] + out
            leaves.append(out)
        else:  # constant
            leaves.append(list(CONSTANTS)[np.random.randint(len(CONSTANTS))])

    # L552
    np.random.shuffle(leaves)

    # insert leaves in the tree
    for pos in range(len(stack) - 1, -1, -1):
        if stack[pos] == None:
            stack = stack[:pos] + [leaves.pop()] + stack[pos + 1:]

    return stack


def write_infix(token, args):
    if token == 'add':
        return f'({args[0]})+({args[1]})'
    elif token == 'sub':
        return f'({args[0]})-({args[1]})'
    elif token == 'mult':
        return f'({args[0]})*({args[1]})'
    elif token == 'div':
        return f'({args[0]})/({args[1]})'
    elif token == 'pow':
        return f'({args[0]})**({args[1]})'
    elif token == 'rac':
        return f'({args[0]})**(1/({args[1]}))'
    elif token == 'abs':
        return f'Abs({args[0]})'
    elif token == 'inv':
        return f'1/({args[0]})'
    elif token == 'pow':
        return f'({args[0]})**{args[1]}'
    elif token in ['sign', 'sqrt', 'exp', 'log', 'sin', 'cos', 'tan']:
        return f'{token}({args[0]})'
    else:
        return token
    raise ValueError(
        f"Unknown token in prefix expression: {token}, with arguments {args}"
    )


def _prefix_to_infix(expr):
    t = expr[0]
    if not isinstance(t, list) and t in OPS:
        args = []
        l1 = expr[1:]
        for _ in range(OPS[t][1]):  # for the arity
            i1, l1 = _prefix_to_infix(l1)
            args.append(i1)
        return write_infix(t, args), l1
    elif t in VARIABLES:
        return t, expr[1:]
    elif t in COEFFICIENTS:
        c = round(np.random.uniform(low=COEFF[0], high=COEFF[1]), ROUND)
        return str(c), expr[1:]
    elif t in CONSTANTS:
        return str(round(getattr(Math, t), ROUND)), expr[1:]
    elif isinstance(t, list):
        return ("".join(t)[3:]), expr[1:]


def prefix_to_infix(expr):
    p, r = _prefix_to_infix(expr)
    if len(r) > 0:
        raise ValueError(
            f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed."
        )
    return f'({p})'


def count_nested_exp(estr):
    stack = []
    count = 0
    max_count = 0
    for v in re.findall('[+-/*//()]|[a-zA-Z0-9]+', estr):
        if v == '(':
            stack.append(v)
        elif v == ')':
            while True:
                x = stack.pop()
                if x in ["exp"]:
                    count -= 1
                if x == '(':
                    break
        else:
            stack.append(v)
            if v in ["exp"]:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count


def is_valid_and_simplify(expr, max_nested_exp=1):
    if len(expr) == 1:
        return False

    # first we check whether there is any variable in the data
    estr = prefix_to_infix(expr)

    # next we check if the expression generated has >max_nested_exp exps
    if count_nested_exp(estr) > max_nested_exp:
        return False

    # next we can check if we see whether any equation can be simplified
    # eg. (((x)+(x))/(4)) = ((x)/(2)) using sympy and then convert the
    # simplified format to brackets to feed to the model
    simp = simplify(estr)
    if type(simp) in [sympy.nan] or isinstance(simp, Float) or isinstance(simp, Integer):
        # sometimes the eqaution can be invalid like x/0
        return False
    
    simp_str = str(simp)
    if not (
        re.findall(r"\(x\)", simp_str) +
        re.findall(r"\(y\)", simp_str) +
        re.findall(r"\(z\)", simp_str) +
        re.findall(r"\(t\)", simp_str)
    ):
        return False

    # sometimes there is a "zoo"
    # print(simp_str)
    if "zoo" in simp_str:
        return False
    
    s2 = simp
    for a in preorder_traversal(simp):
        if isinstance(a, Float):
            s2 = s2.subs(a, round(a, 3))
    return s2


def prepare_expr_string(estr):
    for pat, repl in [
        (str(round(Math.pi, ROUND)), "_pi_"),
        (str(round(Math.e, ROUND)), "_e_"),
        (r"\*\*", "_**_")
    ]:
        estr = re.sub(pat, repl, estr)
    for op in [op for op in list(OPS.keys()) if op in estr]:
	    estr = re.sub(op, f"_{op}_", estr)

    seq = []
    stack = False
    cont_word = ""
    for i,c in enumerate(estr):
        if c == " ": continue
        elif c == "_":
            if stack:
                cont_word += c
                seq.append(cont_word)
                cont_word = ""
                stack = False
            else:
                cont_word += c
                stack = True
        elif stack:
            cont_word += c
        else:
            seq.append(c)
    return seq


@timeout(6)
def create_one_example(
        num_samples,
        dubi, N=3, l=1, p1=1, p2=1,
        leaf_probs=(0.55, 0.1, 0.25, 0.1),
        una_probs=None,
        bi_probs=None
    ):
    expr = generate_random_expression(
        dubi=dubi,
        N=N,
        l=l,
        p1=p1,
        p2=p2,
        leaf_probs=leaf_probs,
        una_probs=una_probs,
        bi_probs=bi_probs)
    simp_expr = is_valid_and_simplify(expr)
    obs = None
    if simp_expr:
        estr = prefix_to_infix(expr)
        variables = list(set(
            re.findall(r"\(x\)", estr) +
            re.findall(r"\(y\)", estr) +
            re.findall(r"\(z\)", estr) +
            re.findall(r"\(t\)", estr)
        ))
        variables = sorted([x[1] for x in variables])
        obs = []
        while len(obs) < num_samples:
            subs = {
                x: round(np.random.uniform(low=-1., high=+1.), ROUND)
                for x in variables
            }
            out = simp_expr.evalf(subs=subs, n=ROUND)
            subs.update({"o": out})
            if "I" in str(out) or str(out) in ["nan"]:  # imaginary numbers like log of a negative number
                continue
            obs.append(subs)
        return obs, prepare_expr_string(str(simp_expr))
    return None, False


def create_one_example_wrapper(maxlen, **kwargs):
    try:
        obs, elist = create_one_example(**kwargs)
        if elist == False:
            raise Exception
        
        # convert to pre-tensor ready data
        elist_tokens = (
            [VOCAB[START_TOKEN], ] + [VOCAB[x] for x in elist] +
            [VOCAB[END_TOKEN], ]
        )[:maxlen + 1]
        elist_tokens += [VOCAB[PAD_TOKEN],]*(maxlen + 1 - len(elist_tokens))
        # print(":::::::", elist, len(elist_tokens))
        attention_mask = [1 for t in elist_tokens if t != VOCAB[PAD_TOKEN]]
        attention_mask += [0,] * (len(elist_tokens) - len(attention_mask))
        decoder_input = {
            "input_ids": np.asarray(elist_tokens),
            "attention_mask": np.asarray(attention_mask)
        }

        encoder_input = {v: [] for v in VARIABLES + ["o"]}
        var_masks = {f"mask_{v}": [] for v in VARIABLES} # this is the variable masker
        for _obs in obs:
            for v in VARIABLES + ["o"]:
                val = _obs.pop(v, 0.)
                encoder_input[v].append(val)
        for k,v in encoder_input.items():
            encoder_input[k] = np.asarray(v)
            if k != "o":
                var_masks[f"mask_{k}"] =  np.asarray([1 if sum(v) != 0 else 0])
        for k, v in var_masks.items():
            var_masks[k] = np.asarray(v)
        encoder_input.update(var_masks)
        return decoder_input, encoder_input
    except Exception as e:
        logging.info(f"Exception: {e}", exc_info = True)
        return create_one_example_wrapper(maxlen, **kwargs)


class O2fDataset(Dataset):
    def __init__(self, args):
        # sanity check
        assert getattr(args, "Nmin")
        assert getattr(args, "Nmax")
        assert getattr(args, "p1min")
        assert getattr(args, "p1max")
        assert getattr(args, "p2min")
        assert getattr(args, "p2max")
        assert getattr(args, "lmin")
        assert getattr(args, "lmax")
        assert getattr(args, "num_samples")
        assert getattr(args, "maxlen")
        assert getattr(args, "dataset_size")

        self.args = args
        self.dataset_size = args.dataset_size

        # list of all the possible permutations of unary-binary distributions so
        # we do not need to compute this thing again and again, but it's fine
        # it takes <1s to do
        self.dubis = {}
        for N in range(args.Nmin, args.Nmax + 1, 1):
            for p1 in range(args.p1min, args.p1max + 1, 1):
                for p2 in range(args.p2min, args.p2max + 1, 1):
                    for l in range(args.lmin, args.lmax + 1, 1):
                        self.dubis[(N, p1, p2, l)] = get_ubi_dist(N, p1, p2, l)

        # quick hacks to force non-singleton expressions, otherwise we have too many duplicates
        self.pn = abs(np.random.normal(size = (args.Nmax + 1 - args.Nmin)))
        self.pn[0] = 0.1
        self.pn = self.pn / self.pn.sum()
        self.pp1 = abs(np.random.normal(size = (args.p1max + 1 - args.p1min)))
        self.pp1 = self.pp1 / self.pp1.sum()
        self.pp2 = abs(np.random.normal(size = (args.p2max + 1 - args.p2min)))
        self.pp2 = self.pp2 / self.pp2.sum()
        self.pl = abs(np.random.normal(size=(args.lmax + 1 - args.lmin)))
        self.pl = self.pl / self.pl.sum()

    @staticmethod
    def _get_N(l, h, p = None):
        return np.random.choice([i for i in range(l, h + 1, 1)], p=p)

    @staticmethod
    def _get_p1(l, h, p = None):
        return np.random.choice([i for i in range(l, h + 1, 1)], p=p)

    @staticmethod
    def _get_p2(l, h, p = None):
        return np.random.choice([i for i in range(l, h + 1, 1)], p=p)

    @staticmethod
    def _get_l(l, h, p=None):
        return np.random.choice([i for i in range(l, h + 1, 1)], p=p)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, *args):
        N = self._get_N(self.args.Nmin, self.args.Nmax, self.pn)
        p1 = self._get_p1(self.args.p1min, self.args.p1max, self.pp1)
        p2 = self._get_p2(self.args.p2min, self.args.p2max, self.pp2)
        l = self._get_l(self.args.lmin, self.args.lmax, self.pl)
        decoder_input, encoder_input = create_one_example_wrapper(
            num_samples=self.args.num_samples,
            dubi=self.dubis[(N, p1, p2, l)],
            N=N, l=l, p1=p1, p2=p2,
            maxlen = self.args.maxlen
        )

        # convert to tensors
        for k, v in decoder_input.items():
            decoder_input[k] = torch.from_numpy(v).long()
        for k,v in encoder_input.items():
            if "mask" in k:
                encoder_input[k] = torch.squeeze(torch.from_numpy(v), -1)
            else:
                # print(v)
                v = v.astype(np.float32)
                encoder_input[k] = torch.unsqueeze(torch.from_numpy(v), -1)
        return encoder_input, decoder_input


class DataConfig():
    Nmin=1
    Nmax=5
    p1min=1
    p1max=3
    p2min=1
    p2max=3
    lmin=1
    lmax=3
    num_samples=40
    dataset_size=10
    maxlen=20
    batch_size = 10


if __name__ == "__main__":
    import sys
    import json
    from uuid import uuid4
    os.makedirs("data", exist_ok=True)

    try:
        mode = sys.argv[1]
    except:
        mode = "tiny"

    if mode == "tiny":
        data_config = SimpleNamespace(
            Nmin=1,
            Nmax=5,
            p1min=1,
            p1max=3,
            p2min=1,
            p2max=3,
            lmin=1,
            lmax=3,
            num_samples=40,
            dataset_size=10,
            maxlen=20,
            batch_size = 10
        )
    elif mode == "large":
        data_config = SimpleNamespace(
            Nmin=1,
            Nmax=5,
            p1min=1,
            p1max=3,
            p2min=1,
            p2max=3,
            lmin=1,
            lmax=3,
            num_samples=40,
            dataset_size=10000,
            maxlen=20,
            batch_size=10000
        )
    else:
        raise ValueError("mode should be in [`tiny`,`large`]")
    ds = O2fDataset(data_config)
    # set_seed(12234)
    encoder = {v: [] for v in VARIABLES + ["o"]}
    decoder = {"input_ids": [], "attention_mask": []}
    start_time = time.time()
    unk_name = str(uuid4())
    print("This run", unk_name)
    for i, (_enc, _dec) in enumerate(DataLoader(ds, batch_size = data_config.batch_size, shuffle = True)):
        # print(f"---> setting seed: {i}")
        # set_seed(i)
        print("---> enc:", {k: (v.size(), v.dtype) for k, v in _enc.items()})
        print("---> dec:", {k: (v.size(), v.dtype) for k, v in _dec.items()})

        # print("--- converting to lists ---")
        enc = {k: v.tolist() for k, v in _enc.items()}
        dec = {k: v.tolist() for k, v in _dec.items()}

        for v in enc:
            encoder.setdefault(v, [])
            encoder[v].extend(enc[v])
        for k in decoder.keys():
            decoder[k].extend(dec[k])
        
        with open(f"data/{unk_name}_sample_{i}.json", "w") as f:
            f.write(json.dumps({
                "encoder": encoder,
                "decoder": decoder
            }))
            show_notification("o2f Data", f"File saving completed at: data/{unk_name}_sample_{i}.json")
            encoder = {v: [] for v in VARIABLES + ["o"]}
            decoder = {"input_ids": [], "attention_mask": []}

    show_notification("o2f Data", f"Script {unk_name} compelte in {time.time() - start_time :.3}s")


