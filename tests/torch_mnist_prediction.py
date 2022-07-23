from __future__ import print_function
import argparse
import random
from datetime import datetime, timedelta
from time import sleep
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "-r", "--run-id", help="run id (default: datetime)", default=datetime.now().strftime("%d%H%M%S%f")[:-5], type=int,)
    parser.add_argument("-s", "--seed", help="seed", default=0, type=int)
    parser.add_argument( "--db", help="database name", default="", type=str,)
    parser.add_argument( "-c", "--comment", help="comment for the experiment (can be used to filter within one db)", default="", type=str,)
    parser.add_argument( "--epochs", help="number of epochs", default=3, type=int)

    parser.add_argument("--step-size", help="step size", default=0.5, type=float)
    parser.add_argument("--batch-size", help="", default=64, type=int)
    parser.add_argument("--test-batch-size", help="", default=1000, type=int)
    parser.add_argument("--momentum", help="", default=0.5, type=float)
    parser.add_argument("--cuda", help="", default=0, type=int)

    # fmt: on

    args = parser.parse_args()
    set_random_seed(args.seed)

    start = timer()
    # load the data
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(14), transforms.Normalize((0.1307,), (0.2801,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = torch.jit.load("mnist_small.pt")
    if args.cuda:
        model.cuda()

    model.eval()
    sample = [x for x, y in train_loader][0][0]
    if args.cuda:
        sample = sample

    print(model(sample).reshape(-1).detach().numpy())

    sample = torch.ones(1,1,14,14).type(torch.FloatTensor)
    print(model(sample).reshape(-1).detach().numpy())

    from IPython import embed; embed()

if __name__ == "__main__":
    main()
