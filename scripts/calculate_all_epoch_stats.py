import click
import json
import pandas
from pathlib import Path

from utils import calculate_statistics, reform_loss_dict


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
def main(in_file, out_dir):
    # Make output dir if it doesn't already exist
    out_dir.mkdir(exist_ok=True)

    # Load loss_dict
    loss_dict = json.loads(in_file.read_text())

    # Get total n epochs
    n_epochs = len(next(iter(loss_dict["train"].values()))["preds"])

    # Calc statistics for each epoch
    for i in range(n_epochs):
        df = reform_loss_dict(loss_dict, pred_epoch=i)
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
        stats_file = out_dir / f"{i}.csv"
        stats_df.to_csv(stats_file, index=False)

        print(f"Finished epoch {i}", flush=True)


if __name__ == "__main__":
    main()
