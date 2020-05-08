"""Copy Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM


# Generator of randomized test sequences
def dataloader(num_batches,
               batch_size,
               seq_width,
               seq_len, test_data):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param seq_len: sequence length for the Priority Sort

    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    if test_data is None:
        test_data = []
    for batch_num in range(num_batches):


        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        while tuple(seq) in test_data:
            seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
            seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the priority
        inp = torch.zeros(seq_len, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        priorities = torch.from_numpy(np.random.random(size=(seq_len, batch_size)) * 2 - 1).unsqueeze(2)
        inp[:, :, seq_width:] = priorities  # 1.0 # priority in our control channel

        sorted_order = np.argsort(priorities, axis=0).squeeze()
        sorted_order = sorted_order.reshape((-1, batch_size))
        #print(inp.shape)
        outp = torch.from_numpy(np.swapaxes(np.stack([inp[:, i][sorted_order[:, i]] for i in range(batch_size)]), 0, 1))
        outp = outp[:,:,:-1]

        yield batch_num+1, inp.float(), outp.float()


@attrs
class PrioritySortParams(object):
    name = attrib(default="priority_sort")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=4,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=8, convert=int)
    sequence_len = attrib(default=20,convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=50000, convert=int)
    batch_size = attrib(default=1, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)
    controller_type = attrib(default="LSTM")
    test_data = attrib(default=None)



#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class PrioritySortModelTraining(object):
    params = attrib(default=Factory(PrioritySortParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 1, self.params.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m, controller_type=self.params.controller_type)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_len, self.params.test_data)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
