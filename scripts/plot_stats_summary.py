import click
import pandas
from pathlib import Path

from utils import plot_stats_summary


@click.command()
@click.option(
    "--stats-file",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "CSV file containing per-split statistics and 95% CI intervals."
        "Column headers should be 'label', 'split', 'statistic', 'value', '95ci_low', "
        "'95ci_high'."
    ),
)
@click.option(
    "-o",
    "--out-file",
    required=True,
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    help="Output plot file.",
)
def main(stats_file, out_file):
    # Load statistics
    stats_df = pandas.read_csv(stats_file)

    plot_stats_summary(stats_df, out_file)


if __name__ == "__main__":
    main()
