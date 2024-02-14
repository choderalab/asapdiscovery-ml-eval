import click
import json
from pathlib import Path

from utils import plot_model_preds_scatter, reform_loss_dict


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
    "--out-dir",
    required=True,
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    help="Output directory to make plots.",
)
def main(in_file, out_dir):
    # Make output dir if it doesn't already exist
    out_dir.mkdir(exist_ok=True)

    for line in in_file.read_text().split("\n"):
        try:
            loss_dict_fn, lab = line.split(",")
        except ValueError:
            if line == "\n":
                continue
            else:
                print(f"Not enough values in line: {line}", flush=True)

        try:
            df = reform_loss_dict(json.loads(Path(loss_dict_fn).read_text()))
        except FileNotFoundError:
            print(f"Couldn't find file {loss_dict_fn}, skipping", flush=True)
            continue
        plot_model_preds_scatter(df, lab, out_dir / f"{lab}.png")
        print(lab, flush=True)


if __name__ == "__main__":
    main()
