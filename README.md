# 2048

## Requirements
* [numba](http://numba.pydata.org/)

## Performance
* Using Numba and totally random play, initial results (total time for 10 000 games):
  * games: 10000, time: 0.626307, avg_score: 1051.317200, avg_move_count: 116.330300
  * Simple MC search (for each move run n random plays, pick move with best total score) - 0.01 s/move (2000 sims per move)
  * MCTS - 0.1 s/move (2000 sims per move), couldn't compile completely

## Performance with neural networks
* Each prediction of the the policy and resulting score of the state takes ~4ms, which is enough to make the MCTS
 over 40 slower than without using the neural network to evaluate leaves.
* This makes it currently impossible to confirm whether the pipeline would provide any useful results

## Network structure
* Input layer
    * Shape (None, 20, 4, 4)
* Convolutional layer
    *  2x2 convolution with 32 filters and stride 1
    *  Batch Normalization
    *  Relu
* 2 Residual layers
    * 2x2 convolution with 32 filters and stride 1
    * Batch Normalization
    * Relu
    * 2x2 convolution with 32 filters and stride 1
    * Batch Normalization
    * Skip link from input
    * Relu
* Policy head
    *  2x2 convolution with 2 filters and stride 1
    *  Batch Normalization
    *  Relu
    *  FC layer w 128 hidden nodes
    *  tanh activation
    *  4 output probabilities
* Value head
    *  2x2 convolution with 1 filters and stride 1
    *  Batch Normalization
    *  Relu
    *  FC layer w 128 hidden nodes
    *  Relu
    *  Output score between 0 and 1