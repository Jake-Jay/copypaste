import torch
import torch.nn as nn


class CopyPasteLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bce_normal = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_copy_paste = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, preds_normal, preds_copy_paste):
        bs = preds_normal.shape[0]

        loss = torch.mean(
            self.bce_normal(preds_normal, torch.zeros(bs, 1)) +
            self.bce_copy_paste(preds_copy_paste, torch.ones(bs, 1)),
        )

        return loss