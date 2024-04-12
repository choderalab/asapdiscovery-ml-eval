import click
import json
import numpy as np
from pathlib import Path

from utils import plot_divergences_with_loss


@click.command()
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Input loss_dict.json file.",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output plot file.",
)
@click.option(
    "-w",
    "--win-size",
    default=[20, 25, 30, 50, 100],
    multiple=True,
    help="Window sizes to calculate divergence for.",
)
def main(in_file, out_file, win_size=[20, 25, 30, 50, 100]):
    loss_dict = json.loads(in_file.read_text())

    train_losses = np.vstack(
        [cpd_d["losses"] for cpd_d in loss_dict["train"].values()]
    ).mean(axis=0)
    val_losses = np.vstack(
        [cpd_d["losses"] for cpd_d in loss_dict["val"].values()]
    ).mean(axis=0)

    plot_divergences_with_loss(train_losses, val_losses, win_size, out_file)


if __name__ == "__main__":
    main()
