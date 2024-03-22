import click
from copy import deepcopy
import json
import multiprocessing as mp
import pandas
from pathlib import Path

from utils import calculate_statistics, reform_loss_dict


def mp_func(loss_dict, pred_epoch, out_fn):
    df = reform_loss_dict(loss_dict, pred_epoch=pred_epoch)
    stats_dict = calculate_statistics(df.loc[df["in_range"] == 0, :])

    stats_rows = [
        ["", sp, stat, stat_d["value"], stat_d["95ci_low"], stat_d["95ci_high"]]
        for sp, sp_d in stats_dict.items()
        for stat, stat_d in sp_d.items()
    ]
    stats_df = pandas.DataFrame(
        stats_rows,
        columns=["label", "split", "statistic", "value", "95ci_low", "95ci_high"],
    )
    stats_df.to_csv(out_fn, index=False)


@click.command()
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="loss_dict JSON file.",
)
@click.option(
    "-o",
    "--out-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory to store all CSV file.",
)
@click.option(
    "-n",
    "--n-workers",
    type=int,
    default=1,
    help="Number of concurrent processes to run.",
)
def main(in_file, out_dir, n_workers=1):
    # Make output dir if it doesn't already exist
    out_dir.mkdir(exist_ok=True)

    # Load loss_dict
    loss_dict = json.loads(in_file.read_text())

    # Get total n epochs
    n_epochs = len(next(iter(loss_dict["train"].values()))["preds"])

    # Set up args for multiprocessing
    mp_args = [(deepcopy(loss_dict), i, out_dir / f"{i}.csv") for i in range(n_epochs)]

    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(mp_func, mp_args)


if __name__ == "__main__":
    main()
