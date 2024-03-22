import click
import json
import pandas
from pathlib import Path

from utils import plot_losses, reform_full_loss_dict


@click.command()
@click.option(
    "-i",
    "--in-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="CSV file containing file paths in the first column and labels in the second.",
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output plot file.",
)
def main(in_file, out_file):
    all_df = []
    converged_dict = {}
    for line in in_file.read_text().split("\n"):
        try:
            loss_dict_fn, lab, *converged_epoch = line.split(",")
        except ValueError:
            if line == "":
                continue
            else:
                print(f"Not enough values in line: {line}", flush=True)

        try:
            df = reform_full_loss_dict(json.loads(Path(loss_dict_fn).read_text()))
        except FileNotFoundError:
            print(f"Couldn't find file {loss_dict_fn}, skipping", flush=True)
            continue

        if len(converged_epoch) > 0:
            converged_dict[lab] = int(converged_epoch[0])

        df["label"] = lab
        all_df.append(df)

    all_df = pandas.concat(all_df, ignore_index=True)
    idx = ["label", "split", "epoch"]
    all_df = all_df.groupby(idx)["loss"].mean().reset_index()

    all_df["trained"] = [
        "trained" if epoch < converged_dict[label] else "not trained"
        for _, (epoch, label) in all_df[["epoch", "label"]].iterrows()
    ]
    print(all_df.head(), flush=True)

    plot_losses(all_df, fn=out_file, splits=["val"])


if __name__ == "__main__":
    main()
