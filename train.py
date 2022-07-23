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

from models.cnn import CNN
from models.fc_net import FCNet


def set_random_seed(seed: int) -> None:
    """
    Seed the different random generators.
    :param seed:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(cuda, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.mse_loss(output, F.one_hot(target, num_classes=10).type(torch.FloatTensor), size_average=False).data
        #test_loss += F.nll_loss(output, target, size_average=False).data
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


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
            "../mnist_data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(14), transforms.Normalize((0.1307,), (0.2801,))]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../mnist_data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(14), transforms.Normalize((0.1307,), (0.2801,))]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
    )

    model = FCNet()
    #from IPython import embed; embed(); exit()
    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.step_size, momentum=args.momentum)

    running_acc = 0;
    for epoch in range(1, args.epochs + 1):
        #train(epoch, args.cuda, model, train_loader, optimizer)
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            #loss = F.nll_loss(output, target)
            loss = F.mse_loss(output, F.one_hot(target, num_classes=10).type(torch.FloatTensor))
            loss.backward()
            optimizer.step()

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).long().cpu().sum()
            running_acc = 0.995 * running_acc + 0.005 * correct / args.batch_size
            if batch_idx % 100 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tRunning Acc: {:.3f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.data,
                        running_acc,
                    )
                )

        test(args.cuda, model, test_loader)

    print("total time: \t", str(timedelta(seconds=timer() - start)))

    model.eval()
    model.cpu()
    sample = [x for x, y in test_loader][0]
    if args.cuda:
        sample = sample
    traced_script_module = torch.jit.trace(model, sample)
    traced_script_module.save("mnist_small.pt")


if __name__ == "__main__":
    main()
