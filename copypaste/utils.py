import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Visualisation Utilities
# ------------------------------------------------------------------------------
def inspect(img):
    fig = plt.figure()
    plt.imshow(img.movedim(0,-1))

    return fig

def inspect_pair(img: torch.Tensor, cp: torch.Tensor) -> plt.figure():
    fig, axs = plt.subplots(1, 2)

    axs[0].imshow(img.moveaxis(0,-1))
    axs[0].set_title('Original')
    axs[1].imshow(cp.moveaxis(0,-1))
    axs[1].set_title('Copy Paste')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    return fig