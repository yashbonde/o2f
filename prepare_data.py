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
import signal
from functools import wraps, partial

import sympy
from sympy import simplify, Float, Integer, preorder_traversal
from sympy.solvers import solvers
import os
from maths import Math
import numpy as np
from argparse import ArgumentParser
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader

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
VOCAB_TOKENS += [START_TOKEN + PAD_TOKEN + END_TOKEN]  # language modelling
VOCAB_TOKENS += ["o"] # ??
VOCAB = {t: i for i, t in enumerate(VOCAB_TOKENS)}

# ---- util ---- #
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):

    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator

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


@timeout(10)
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
            if "I" in str(out) or str(out) in ["nan"]:  # imaginary numbers like log of a negative number
                continue
            obs.append((subs, out))
    return obs, prepare_expr_string(str(simp_expr))


class o2fDataset(Dataset):
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
        assert getattr(args, "sample_seq")
        assert getattr(args, "maxlen")
        assert getattr(args, "dataset_size")

        self.args = args

        # list of all the possible permutations of unary-binary distributions so
        # we do not need to compute this thing again and again
        self.dubis = {}
        start_time = time.time()
        for N in range(args.Nmin, args.Nmax + 1, 1):
            for p1 in range(args.p1min, args.p1max + 1, 1):
                for p2 in range(args.p2min, args.p2max + 1, 1):
                    for l in range(args.lmin, args.lmax + 1, 1):
                        self.dubis[(N, p1, p2, l)] = get_ubi_dist(N, p1, p2, l)
        print("Loading distributions took:", start_time - time.time(), "s")

    def _get_N(self):
        return np.random.radnint(low=self.args.Nmin, high=self.args.Nmax)

    def _get_p1(self):
        return np.random.randint(low=self.args.p1min, high=self.args.p1max)

    def _get_p2(self):
        return np.random.randint(low=self.args.p2min, high=self.args.p2max)

    def _get_l(self):
        return np.random.randint(low=self.args.lmin, high=self.args.lmax)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, *args):
        N, p1, p2, l = self._get_N(), self._get_p1(), self._get_p2(), self._get_l()
        obs, elist = create_one_example(
            num_samples=args.num_samples, dubi=self.dubis[(N, p1, p2, l)],
            N=N, l=l, p1=p1, p2=p2
        )
        while obs is None:
            obs, elist = create_one_example(
                num_samples=args.num_samples, dubi=self.dubis[(N, p1, p2, l)],
                N=N, l=l, p1=p1, p2=p2
            )

        # convert to tokens for decoder
        elist_tokens = (
            (VOCAB[START_TOKEN],) + (VOCAB[x] for x in elist) +
            (VOCAB[END_TOKEN],)
        )[:self.args.maxlen + 1]
        elist_tokens += tuple([VOCAB[PAD_TOKEN],]*(self.args.maxlen - len(elist_tokens)))
        attention_mask = [1 for t in elist_tokens if t != VOCAB[PAD_TOKEN]]
        attention_mask += [0,] * (len(elist_tokens) - len(attention_mask))
        decoder_input = {
            "tokens": torch.from_numpy(np.asarray(elist_tokens)).long(),
            "attention_mask": torch.from_numpy(np.asarray(attention_mask)).long()
        }

        # convert observation to input for model
        encoder_input = {
            "x": [], # var-x
            "y": [], # var-y
            "z": [], # var-z
            "t": [], # var-t
            "o": [] # output
        }
        

        # for _ in range(10):
        #     try:
        #         obs, estr = create_one_example(num_samples=30, dubi=dubi)
        #         if obs is not None:
        #             # for x in obs:
        #             #     print(list(x[0].values()), "==>", x[1])
        #             # print("EXPR:", estr, [VOCAB[x] for x in estr])
        #     except Exception as e:
        #         pass

        # infix = generate_random_expression(N, l, p1, p2)
        # while not is_valid(infix):
        #     infix = generate_random_expression(N, l, p1, p2)
        # o = [generate_obs(infix, pt=True) for _ in range(args.sample_seq)]

        return o


# if __name__ == "__main__":
#     print(VOCAB)

#     dubi = get_ubi_dist()
#     # for _ in trange(100, ncols = 100):
#     for _ in range(100):
#         try:
#             obs, estr = create_one_example(num_samples=30, dubi=dubi)
#             if obs is not None:
#                 # for x in obs:
#                 #     print(list(x[0].values()), "==>", x[1])
#                 print("EXPR:", estr, [VOCAB[x] for x in estr])
#         except Exception as e:
#             print("---->", e)
#         print()


if __name__ == "__main__":
    data_config = SimpleNamespace(
        Nmin=1,
        Nmax=5,
        p1min=1,
        p1max=3,
        p2min=1,
        p2max=3,
        lmin=1,
        lmax=3,
        sample_seq=40,
        dataset_size=1000,
        maxlen=20
    )
    dl = DataLoader(**data_config)
