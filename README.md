# Fast 2048 in python

Trying to implement the fastest 2048 simulation in python, in order to use it as a machine learning base.

Currently the basic game is JIT compiled using numba. These are the performance characteristics (time per single run):

|run      | JITed  | Pure Python |
|---------|--------|-------------|
|1        | 3.4s   | 3.3ms       |
|2-1001   | 0.13ms | 2.7ms       |
|1002-2001| 0.13ms | 2.7ms       |

As you can see, the first run of the JITed version takes a considerable length of time, but afterwards the performance is just stellar (only requires 5% of the time). This is a twenty-fold improvement.

This means, that after approximately 1350 runs, the JITed version has broken even, and after 10000 iterations, about 22 seconds will be saved (the JITed version will take just about 4.7s, while the pure python one takes 27s).