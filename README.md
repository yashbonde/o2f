# o2f: Observation to Formula

Can AI predict formulas by simply looking at raw observational data!

## [Generating Expressions](./generating_expressions.md)

For expression generation I used the method mentioned in [Deep Learning for Symbolic Mathematics](http://arxiv.org/abs/1912.01412) and the script [here](https://github.com/facebookresearch/SymbolicMathematics/blob/master/src/envs/char_sp.py). However to make things simpler to understand, I have rewrote the code to be much smaller and easily understandable. Though some helper functions have been taken as is.

There is some extra engineering requried to generate obeservations, which I tried to take from the above mentioned code base but eventually wrote my own simpler thing.
