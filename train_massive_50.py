#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""Training for the Copy Task in Neural Turing Machines."""

import argparse
import json
import logging
import time
import random
import re
import sys

import attr
import argcomplete
import torch
import numpy as np


LOGGER = logging.getLogger(__name__)


from tasks.copytask import CopyTaskModelTraining, CopyTaskParams
from tasks.repeatcopytask import RepeatCopyTaskModelTraining, RepeatCopyTaskParams
from tasks.priority_sort import PrioritySortModelTraining, PrioritySortParams, dataloader # dirty hack for train/test split

TASKS = {
    'copy': (CopyTaskModelTraining, CopyTaskParams),
    'repeat-copy': (RepeatCopyTaskModelTraining, RepeatCopyTaskParams),
    'priority-sort':(PrioritySortModelTraining, PrioritySortParams)
}

#torch.set_default_tensor_type('torch.cuda.FloatTensor')
# Default values for program arguments
RANDOM_SEED = 1000
REPORT_INTERVAL = 200
CHECKPOINT_INTERVAL = 1000


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000


def init_seed(seed=None):
    """Seed the RNGs for predicatability/reproduction purposes."""
    if seed is None:
        seed = int(get_ms() // 1000)

    LOGGER.info("Using seed=%d", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def progress_clean():
    """Clean the progress bar."""
    print("\r{}".format(" " * 80), end='\r')


def progress_bar(batch_num, report_interval, last_loss):
    """Prints the progress until the next report."""
    progress = (((batch_num-1) % report_interval) + 1) / report_interval
    fill = int(progress * 40)
    print("\r[{}{}]: {} (Loss: {:.4f})".format(
        "=" * fill, " " * (40 - fill), batch_num, last_loss), end='')


def save_checkpoint(net, name, args, batch_num, losses, costs, seq_lengths,
                    controller_size, controller_layers, seq_len):#,
                    #test_loss_usual, test_loss_big):
    progress_clean()

    basename = "{}/{}-{}-{}-{}-{}-batch-{}".format(args.checkpoint_path, name, controller_size, controller_layers, seq_len , args.seed, batch_num)
    model_fname = basename + ".model"
    LOGGER.info("Saving model checkpoint to: '%s'", model_fname)
    torch.save(net.state_dict(), model_fname)

    # Save the training history
    train_fname = basename + ".json"
    LOGGER.info("Saving model training history to '%s'", train_fname)
    content = {
        "loss": losses,
        "cost": costs,
        "seq_lengths": seq_lengths
    }
    open(train_fname, 'wt').write(json.dumps(content))


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def train_batch(net, criterion, optimizer, X, Y):
    """Trains a single batch."""
    optimizer.zero_grad()
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    for i in range(inp_seq_len):
        net(X[i])#.cuda())

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], _ = net()

    loss = criterion(y_out, Y)#.cuda())
    loss.backward()
    clip_grads(net)
    optimizer.step()

    y_out_binarized = y_out.clone().data.cpu()
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    return loss.item(), cost.item() / batch_size


def evaluate(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    y_out = torch.zeros(Y.size())
    for i in range(outp_seq_len):
        y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss,#,.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result


def train_model(model, args):
    num_batches = model.params.num_batches
    batch_size = model.params.batch_size
    #test_data = model.params.test_data


    LOGGER.info("Training model for %d batches (batch_size=%d)...",
                num_batches, batch_size)
    #model.net#.cuda()
    # test_data_bigger =  {tuple(x):tuple(y) for _,x,y in list(dataloader(500, 1,
    #                           model.params.sequence_width,
    #                           model.params.sequence_len*2, None))}

    losses = []
    costs = []
    seq_lengths = []
    start_ms = get_ms()
    #test_loss = []
    #test_loss_bigger = []

    for batch_num, x, y in model.dataloader:
        loss, cost = train_batch(model.net, model.criterion, model.optimizer, x, y)
        losses += [loss]
        costs += [cost]
        seq_lengths += [y.size(0)]

        # if model.params.name=='priority_sort':
        #     test_loss+=evaluate(model.net, model.criterion, [x for x in test_data.keys()], [y for y in test_data.values()])
        #     test_loss_bigger+=evaluate(model.net, model.criterion, [x for x in test_data_bigger.keys()], [y for y in test_data_bigger.values()])
        # Update the progress bar
        progress_bar(batch_num, args.report_interval, loss)

        # Report
        if batch_num % args.report_interval == 0:
            mean_loss = np.array(losses[-args.report_interval:]).mean()
            mean_cost = np.array(costs[-args.report_interval:]).mean()
            mean_time = int(((get_ms() - start_ms) / args.report_interval) / batch_size)
            progress_clean()
            LOGGER.info("Batch %d Loss: %.6f Cost: %.2f Time: %d ms/sequence",
                        batch_num, mean_loss, mean_cost, mean_time)
            start_ms = get_ms()

        # Checkpoint
        if (args.checkpoint_interval != 0) and (batch_num % args.checkpoint_interval == 0):
            save_checkpoint(model.net, model.params.name, args,
                            batch_num, losses, costs, seq_lengths,
                            model.params.controller_size, model.params.controller_layers, model.params.sequence_len)#,
                            #test_loss, test_loss_bigger)

    LOGGER.info("Done training.")


def init_arguments():
    parser = argparse.ArgumentParser(prog='train.py')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED, help="Seed value for RNGs")
    parser.add_argument('--task', action='store', choices=list(TASKS.keys()), default='priority-sort',
                        help="Choose the task to train (default: priority-sort)")
    parser.add_argument('-p', '--param', action='append', default=[],
                        help='Override model params. Example: "-pbatch_size=4 -pnum_heads=2"')
    parser.add_argument('--checkpoint-interval', type=int, default=CHECKPOINT_INTERVAL,
                        help="Checkpoint interval (default: {}). "
                             "Use 0 to disable checkpointing".format(CHECKPOINT_INTERVAL))
    parser.add_argument('--checkpoint-path', action='store', default='./',
                        help="Path for saving checkpoint data (default: './')")
    parser.add_argument('--report-interval', type=int, default=REPORT_INTERVAL,
                        help="Reporting interval")

    argcomplete.autocomplete(parser)

    args = parser.parse_args()
    args.checkpoint_path = args.checkpoint_path.rstrip('/')

    return args


def update_model_params(params, update):
    """Updates the default parameters using supplied user arguments."""

    update_dict = {}
    for p in update:
        m = re.match("(.*)=(.*)", p)
        if not m:
            LOGGER.error("Unable to parse param update '%s'", p)
            sys.exit(1)

        k, v = m.groups()
        update_dict[k] = v

    try:
        params = attr.evolve(params, **update_dict)
    except TypeError as e:
        LOGGER.error(e)
        LOGGER.error("Valid parameters: %s", list(attr.asdict(params).keys()))
        sys.exit(1)
    #print('params', params)
    return params

def init_model(args):
    LOGGER.info("Training for the **%s** task", args.task)

    model_cls, params_cls = TASKS[args.task]
    params = params_cls()

    params = update_model_params(params, args.param)
    LOGGER.info(params)

    # dirty hack for train/test data
    if params.name == 'priority_sort':
        test_data = {tuple(x):tuple(y) for _,x,y in list(dataloader(500, 1,
                              params.sequence_width,
                              params.sequence_len, None))}
        import pickle

        with open("test_data-contr_size-{}-contr_layers-{}-seqlen-{}.pickle".format(params.controller_size, params.controller_layers, params.sequence_len),'wb') as f:
            pickle.dump(test_data, f)

        test_data_big = {tuple(x): tuple(y) for _, x, y in list(dataloader(500, 1,
                                                                       params.sequence_width,
                                                                       params.sequence_len, None))}
        with open("test_data_BIG-contr_size-{}-contr_layers-{}-seqlen-{}.pickle".format(params.controller_size,
                                                                                    params.controller_layers,
                                                                                    params.sequence_len*2), 'wb') as f:
            pickle.dump(test_data_big, f)

        # print(list(dataloader(4, 1,
        #                       params.sequence_width,
        #                       params.sequence_len, None))[1])
        #print(list(test_data.keys())[0])

        params = attr.evolve(params,test_data = test_data)
    # end of dirty hack


    model = model_cls(params=params)

    return model


def init_logging():
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] [%(name)s]  %(message)s',
                        level=logging.DEBUG)


def main():
    init_logging()

    # Initialize arguments
    args = init_arguments()
    # Initialize random
    init_seed(args.seed)
    print('INITIALIZED')

    # Initialize the model
    for controller_size in [50]:
        for seq_len in [7]:
            args.param = ['controller_size={}'.format(controller_size),'sequence_len={}'.format(seq_len),"controller_type=Transformer"]
            print(args)

            model = init_model(args)

            LOGGER.info("Total number of parameters: %d", model.net.calculate_num_params())
            train_model(model, args)



if __name__ == '__main__':
    main()
