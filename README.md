## This is repo for final project in Neural Program Learning.
### Danylo Honcharov
### It is forked from [this repository](https://github.com/loudinthecloud/pytorch-ntm)
Usage is very similar to the original repo:
Execute ./train.py

```
usage: train.py [-h] [--seed SEED] [--task {copy,repeat-copy}] [-p PARAM]
                [--checkpoint-interval CHECKPOINT_INTERVAL]
                [--checkpoint-path CHECKPOINT_PATH]
                [--report-interval REPORT_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Seed value for RNGs
  --task {copy,repeat-copy}
                        Choose the task to train (default: copy)
  -p PARAM, --param PARAM
                        Override model params. Example: "-pbatch_size=4
                        -pnum_heads=2"
  --checkpoint-interval CHECKPOINT_INTERVAL
                        Checkpoint interval (default: 1000). Use 0 to disable
                        checkpointing
  --checkpoint-path CHECKPOINT_PATH
                        Path for saving checkpoint data (default: './')
  --report-interval REPORT_INTERVAL
                        Reporting interval
```

There are new model params - "controller_type", which is "LSTM" by default, but can be "Transformer".

Also there are new task for model evaluation - 'priority-sort'. Currently 'copy' task may be broken and will be fixed in the future.
To install all the dependencies:

1. install everything from requirements.txt


To run training of the model on priority-sort with Transformer (priority-sort task is set by default):

1. run `mkdir notebook/sort` to create new directory for checkpoints;
2. run `train.py --seed 1  --checkpoint-interval 500 --checkpoint-path .notebooks/sort -pcontroller_type=Transformer`

Test dataset will be saved in the current directory in the pickle file with format "test_data-contr_size-{}-contr_layers-{}-seqlen-{}.pickle" with parameters of model instead of {} symbols.

This code is tested on the Linux systems.

Also I get excited and almost finished QRNN controller (which I wasn't supposed to do), but it wasn't tested and may be broken.

All notebooks, necessary for results reproducing can be found in the notebooks directory (though they are not polished yet).
train_massive_50.py,train_massive_100.py, train_massive_200.py were used for the parallel training and should be ignored (and they not very different from train.py).

## Readme from original Pytorch-NTM repo below:

# PyTorch Neural Turing Machine (NTM)

PyTorch implementation of [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (NTM).

An **NTM** is a memory augumented neural network (attached to external memory) where the interactions with the external memory (address, read, write) are done using differentiable transformations. Overall, the network is end-to-end differentiable and thus trainable by a gradient based optimizer.

The NTM is processing input in sequences, much like an LSTM, but with additional benfits: (1) The external memory allows the network to learn algorithmic tasks easier (2) Having larger capacity, without increasing the network's trainable parameters.

The external memory allows the NTM to learn algorithmic tasks, that are much harder for LSTM to learn, and to maintain an internal state much longer than traditional LSTMs.

## A PyTorch Implementation

This repository implements a vanilla NTM in a straight forward way. The following architecture is used:

![NTM Architecture](./images/ntm.png)

### Features
* Batch learning support
* Numerically stable
* Flexible head configuration - use X read heads and Y write heads and specify the order of operation
* **copy** and **repeat-copy** experiments agree with the paper

***

## Copy Task

The **Copy** task tests the NTM's ability to store and recall a long sequence of arbitrary information. The input to the network is a random sequence of bits, ending with a delimiter. The sequence lengths are randomised between 1 to 20.

### Training

Training convergence for the **copy task** using 4 different seeds (see the [notebook](./notebooks/copy-task-plots.ipynb) for details)

![NTM Convergence](./images/copy-train.png)

 The following plot shows the cost per sequence length during training. The network was trained with `seed=10` and shows fast convergence. Other seeds may not perform as well but should converge in less than 30K iterations.

![NTM Convergence](./images/copy-train2.png)

### Evaluation

Here is an animated GIF that shows how the model generalize. The model was evaluated after every 500 training samples, using the target sequence shown in the upper part of the image. The bottom part shows the network output at any given training stage.

![Copy Task](./images/copy-train-20-fast.gif)

The following is the same, but with `sequence length = 80`. Note that the network was trained with sequences of lengths 1 to 20.

![Copy Task](./images/copy-train-80-fast.gif)

***
## Repeat Copy Task

The **Repeat Copy** task tests whether the NTM can learn a simple nested function, and invoke it by learning to execute a __for loop__. The input to the network is a random sequence of bits, followed by a delimiter and a scalar value that represents the number of repetitions to output. The number of repetitions, was normalized to have zero mean and variance of one (as in the paper). Both the length of the sequence and the number of repetitions are randomised between 1 to 10.

### Training

Training convergence for the **repeat-copy task** using 4 different seeds (see the [notebook](./notebooks/repeat-copy-task-plots.ipynb) for details)

![NTM Convergence](./images/repeat-copy-train.png)

### Evaluation

The following image shows the input presented to the network, a sequence of bits + delimiter + num-reps scalar. Specifically the sequence length here is eight and the number of repetitions is five.

![Repeat Copy Task](./images/repeat-copy-ex-inp.png)

And here's the output the network had predicted:

![Repeat Copy Task](./images/repeat-copy-ex-outp.png)

Here's an animated GIF that shows how the network learns to predict the targets. Specifically, the network was evaluated in each checkpoint saved during training with the same input sequence.

![Repeat Copy Task](./images/repeat-copy-train-10.gif)

## Installation

The NTM can be used as a reusable module, currently not packaged though.

1. Clone repository
2. Install [PyTorch](http://pytorch.org/)
3. pip install -r requirements.txt

## Usage

Execute ./train.py

```
usage: train.py [-h] [--seed SEED] [--task {copy,repeat-copy}] [-p PARAM]
                [--checkpoint-interval CHECKPOINT_INTERVAL]
                [--checkpoint-path CHECKPOINT_PATH]
                [--report-interval REPORT_INTERVAL]

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Seed value for RNGs
  --task {copy,repeat-copy}
                        Choose the task to train (default: copy)
  -p PARAM, --param PARAM
                        Override model params. Example: "-pbatch_size=4
                        -pnum_heads=2"
  --checkpoint-interval CHECKPOINT_INTERVAL
                        Checkpoint interval (default: 1000). Use 0 to disable
                        checkpointing
  --checkpoint-path CHECKPOINT_PATH
                        Path for saving checkpoint data (default: './')
  --report-interval REPORT_INTERVAL
                        Reporting interval
```
