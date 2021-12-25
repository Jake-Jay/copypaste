import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from copypaste.utils import inspect, inspect_pair
from copypaste.data import MVTecDataset
from copypaste.transforms import FancyPCA
from copypaste.models import AnomolyRepresentationLearner
from copypaste.losses import CopyPasteLoss


def train_epoch(train_dict, writer):
    
    optimizer = train_dict['optimizer']
    dl = train_dict['dataloader']
    criterion = train_dict['criterion']

    running_loss = 0.0

    for i, data in enumerate(dl):
        img, cp = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_norm = model(img)
        pred_cp = model(cp)

        loss = criterion(pred_norm, pred_cp)
        loss.backward()
        writer.add_scalar('Loss/train', loss, epoch*n_batches + i)
        
        optimizer.step()

        running_loss += loss.item()
    
    return running_loss / i

def parse_cli_arguments() -> dict:
    parser = argparse.ArgumentParser()

    # train setup
    parser.add_argument("--dataset", 
                        help="the intermediate dataset will be stored/loaded here",
                        default='/Users/marcobertolini/Documents/Arbeit/data/pathology/tggates/segmentation/')
    parser.add_argument("--workers", default=0, type=int, help="the number of workers used by the loaders")
    parser.add_argument('--save_dir', action='store', dest='save_dir', default="results", type=str,
                        help="Where the results should be saved. Defaults to results/")
    
    # model hyperparams
    parser.add_argument('--lr', action='store', dest='lr', default=0.01, type=float,
                        help="Initial learning rate for training. Defaults to 0.01.")
    parser.add_argument("--batch_size", default=8, type=int, help="the batch size for the loaders")
    parser.add_argument("--n_epochs", default=10, type=int, help="train for this many epochs")
    parser.add_argument('--transform', dest='use_transform', action='store_true', 
                        help="Use geometric transformations")
    
    # miscellaneous

    # check command line arguments
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_cli_arguments()

    # path = "/Users/jake/bayer/copypaste/data/toothbrush/train"
    options = {
        'area_ratio_range': (0.01, 0.05)
    }
    transforms = [FancyPCA()]

    ds = MVTecDataset(
        args.dataset, 
        copypaste=True, 
        transforms=transforms, **options
    )

    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        num_workers=2
    )

    model = AnomolyRepresentationLearner()
    criterion = CopyPasteLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    training_dict = {
        'dataloader': dl,
        'model': model,
        'criterion': criterion,
        'optimizer': optimizer,
    }

    n_batches = len(ds) / args.batch_size

    writer = SummaryWriter()

    for epoch in range(2):
        running_loss = 0.0

        avg_train_loss = train_epoch(training_dict, writer)

    print('Finished Training')


