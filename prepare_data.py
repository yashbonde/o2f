"""~script to generate data for o2f model~
dataloader object for o2f model, saving the data doesn't really make sense
because of large dataset
29.08.2020 - @yashbonde

This is inspired from FAIR paper: Deep Learning for Symbolic Mathematics
http://arxiv.org/abs/1912.01412

For more details read: generate_expressions.md
"""
import os
from maths import Math
import numpy as np
from argparse import ArgumentParser
from types import SimpleNamespace

from torch.util.data import Dataset, Dataloader

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
} # tuple of <op, arity>

VARIABLES = [i for i in "xyzt"]
COEFFICIENTS = [f"a{i}" for i in range(10)]
CONSTANTS = ["pi", "E"]

ROUND = 3
COEFF = [0, +10.]

UNA_OPS = [k for k, v in OPS.items() if v[1] == 1]
BIN_OPS = [k for k, v in OPS.items() if v[1] == 2]

# ---- functions ---- #
def get_ubi_dist(N = 3, p1 = 1, p2 = 1, l = 1):
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
        N = 3,
        l = 1,
        p1 = 1,
        p2 = 1,
        leaf_probs = (0.55, 0.1, 0.25, 0.1),
        una_probs = None,
        bi_probs = None
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
    e = 1 # number of empty leaves
    l_leaves = 0 # left leaves - None state reserved for leaves
    t_leaves = 1 # total number of leaves (for sanity)
    stack = [None]
    
    # start making tree
    for _n in range(N, 0, -1):
        #L523 -- get the next operator, arity and position
        
        #L474 -- sample a position of the next node (unary-binary case)
        # sample a position in {0, ..., e-1}, along with arity
        probs = []
        for k in range(e):
            probs.append((l**k) * p1 * dubi[e - k][_n - 1])
        for k in range(e):
            probs.append((l**k) * p2 * dubi[e - k + 1][_n - 1])
        probs = [p / dubi[e][_n] for p in probs]
        probs = np.array(probs, dtype = np.float64)
        k = np.random.choice(2 * e, p = probs)
        a = 1 if k < e else 2 # arity
        k = k % e # position
        #L491
        
        #L524 -- now we have sampled the position and arity we will
        # sample the operator
        if a == 1:
            op = np.random.choice(UNA_OPS, p = una_probs)
        else:
            op = np.random.choice(BIN_OPS, p = bi_probs)
        
        e = e + OPS[op][1] - 1 - k # created empty nodes - skipped future leaves
        t_leaves = t_leaves + OPS[op][1] - 1 # update total no. of leaves
        l_leaves = l_leaves + k # update no. of left leaves
        
        # update the tree
        pos = [i for i,v in enumerate(stack) if v is None][l_leaves]
        stack = stack[:pos] + [op] + [None for _ in range(OPS[op][1])] + stack[pos + 1:]

    #L545 -- create leaves
    leaves = []
    for _ in range(t_leaves):
        #L498
        leaf_type = np.random.choice(4, p = leaf_probs)
        if leaf_type == 0: # variable
            leaves.append(list(VARIABLES)[np.random.randint(len(VARIABLES))])
        elif leaf_type == 1: # coefficient
            leaves.append(list(COEFFICIENTS)[np.random.randint(len(COEFFICIENTS))])
        elif leaf_type == 2: # integer
            c = np.random.randint(low = COEFF[0], high = COEFF[1])
            c = c if np.random.randint(2) == 0 else -c
            out = [x for x in str(c)]
            if out[0] == "-":
                out[0] = "INT-"
            else:
                out = ["INT"] + out
            leaves.append(out)
        else: # constant
            leaves.append(list(CONSTANTS)[np.random.randint(len(CONSTANTS))])

    #L552
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
    raise ValueError(f"Unknown token in prefix expression: {token}, with arguments {args}")


def _prefix_to_infix(expr): 
    t = expr[0]
    if not isinstance(t, list) and t in OPS:
        args = []
        l1 = expr[1:]
        for _ in range(OPS[t][1]): # for the arity
            i1, l1 = _prefix_to_infix(l1)
            args.append(i1)
        return write_infix(t, args), l1
    elif t in VARIABLES:
        return t, expr[1:]
    elif t in COEFFICIENTS:
        c = round(np.random.uniform(low = COEFF[0], high = COEFF[1]), ROUND)
        return str(c), expr[1:]
    elif t in CONSTANTS:
        return str(round(getattr(Math, t), ROUND)), expr[1:]
    elif isinstance(t, list):
        return ("".join(t)[3:]), expr[1:]


def prefix_to_infix(expr):
    p, r = _prefix_to_infix(expr)
    if len(r) > 0:
        raise ValueError(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
    return f'({p})'


def is_valid(expr):
    pass


def generate_obs(expr, pt = True):

    if pt:
        # return tensors
        pass
    pass




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
        assert getattr(args, "dataset_size")

        self.args = args

        # list of all the possible permutations of unary-binary distributions so
        # we do not need to compute this thing again and again
        self.dubis = {}
        for N in range(args.Nmin, args.Nmax + 1, 1):
            for p1 in range(args.p1min, args.p1max + 1, 1):
                for p2 in range(args.p2min, args.p2max + 1, 1):
                    for l in range(args.lmin, args.lmax + 1, 1):
                        self.dubis[(N, p1, p2, l)] = get_ubi_dist(N, p1, p2, l)

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
        infix = generate_random_expression(N, l, p1, p2)
        while not is_valid(infix):
            infix = generate_random_expression(N, l, p1, p2)
        
        o = [generate_obs(infix, pt = True) for _ in range(args.sample_seq)]

        return o

        





# # ---- script ---- #
# if __name__ == "__main__":
#     args = ArgumentParser("script to prepare data for o2f")
#     args.add_argument("--samples", default = int(1e6), type = int, help = "number of samples to prepare")
#     args.add_argument("--folder", type = str, default = "o2f_data", help = "folder to dump")

#     # generations arguments
#     args.add_argument("--Nmin", type = int, default = 1, choices = [i for i in range(9)], help = "what are the minimum number of operators to include in the expression generation engine")
#     args.add_argument("--Nmax", type = int, defualt = 4, choices = [i for i in range(1, 10)], help = "what are the maximum number of operators to include in the expression generation engine")

#     args.add_argument("--p1min", type = int, default = 1, choices = [i for i in range(9)], help = "what are the minimum number of unary operators to include in the expression generation engine")
#     args.add_argument("--p1max", type = int, defualt = 4, choices = [i for i in range(1, 10)], help = "what are the maximum number of unary operators to include in the expression generation engine")

#     args.add_argument("--p2min", type = int, default = 1, choices = [i for i in range(9)], help = "what are the minimum number of binary operators to include in the expression generation engine")
#     args.add_argument("--p2max", type = int, defualt = 4, choices = [i for i in range(1, 10)], help = "what are the maximum number of binary operators to include in the expression generation engine")

#     args.add_argument("--lmin", type=int, default=1, choices=[i for i in range(9)], help="what are the minimum number of leaves to include in the expression generation engine")
#     args.add_argument("--lmax", type=int, defualt=4, choices=[i for i in range(1, 10)], help="what are the maximum number of leaves to include in the expression generation engine")

#     args = args.parse_args()

#     os.makedirs(args.folder, exist_ok=True)

#     # make the possible combinations
