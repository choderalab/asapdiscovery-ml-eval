import click
import json
import pandas
from pathlib import Path

from utils import calculate_statistics, plot_model_preds_scatter, reform_loss_dict


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
@click.option(
    "--stats-file",
    required=False,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "CSV file containing per-split statistics and 95% CI intervals."
        "Column headers should be 'label', 'split', 'statistic', 'value', '95ci_low', "
        "'95ci_high'."
    ),
)
def main(in_file, out_dir, stats_file):
    # Make output dir if it doesn't already exist
    out_dir.mkdir(exist_ok=True)

    # Load stats
    if stats_file and stats_file.exists():
        stats_df = pandas.read_csv(stats_file)
        # Parse df -> dict of label: split: stat: {value, ci}
        stats_dict = {
            lab: (
                g.groupby("split")[["statistic", "value", "95ci_low", "95ci_high"]]
                .apply(lambda x: x.set_index("statistic").to_dict(orient="index"))
                .to_dict()
            )
            for lab, g in stats_df.groupby("label")
        }
        calc_stats = False
    else:
        if stats_file:
            calc_stats = True
        else:
            calc_stats = False
        stats_dict = {}

    for line in in_file.read_text().split("\n"):
        try:
            loss_dict_fn, lab = line.split(",")
        except ValueError:
            if line == "":
                continue
            else:
                print(f"Not enough values in line: {line}", flush=True)

        try:
            df = reform_loss_dict(json.loads(Path(loss_dict_fn).read_text()))
        except FileNotFoundError:
            print(f"Couldn't find file {loss_dict_fn}, skipping", flush=True)
            continue

        # Calculate state if needed (only if we didn't load them and we do
        #  want to save them)
        if calc_stats and stats_file:
            stats_dict[lab] = calculate_statistics(df.loc[df["in_range"] == 0, :])

        plot_model_preds_scatter(
            df, lab, out_dir / f"{lab}.png", stats_dict.get(lab, {})
        )
        print(lab, flush=True)

    # Save stats_dict if needed
    if calc_stats and stats_file:
        stats_rows = [
            [lab, sp, stat, stat_d["value"], stat_d["95ci_low"], stat_d["95ci_high"]]
            for lab, lab_d in stats_dict.items()
            for sp, sp_d in lab_d.items()
            for stat, stat_d in sp_d.items()
        ]
        stats_df = pandas.DataFrame(
            stats_rows,
            columns=["label", "split", "statistic", "value", "95ci_low", "95ci_high"],
        )
        stats_df.to_csv(stats_file, index=False)


if __name__ == "__main__":
    main()
