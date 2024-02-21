from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import bootstrap, kendalltau, spearmanr
import seaborn as sns


def calculate_statistics(df):
    stats_dict = {"train": {}, "val": {}, "test": {}}

    for i, sp in enumerate(["train", "val", "test"]):
        df_tmp = df.loc[df["split"] == sp, ["target", "pred"]]

        # Calculate MAE and bootstrapped confidence interval
        mae = (df_tmp["target"] - df_tmp["pred"]).abs().mean()
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: np.abs(target - pred).mean(),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["mae"] = {
            "value": mae,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

        # Calculate RMSE and bootstrapped confidence interval
        rmse = np.sqrt((df_tmp["target"] - df_tmp["pred"]).pow(2).mean())
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: np.sqrt(np.power(target - pred, 2).mean()),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["rmse"] = {
            "value": rmse,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

        # Calculate Spearman r and bootstrapped confidence interval
        sp_r = spearmanr(df_tmp["target"], df_tmp["pred"]).statistic
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: spearmanr(target, pred).statistic,
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["sp_r"] = {
            "value": sp_r,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

        # Calculate Kendall's tau and bootstrapped confidence interval
        tau = kendalltau(df_tmp["target"], df_tmp["pred"]).statistic
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: kendalltau(target, pred).statistic,
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        stats_dict[sp]["tau"] = {
            "value": tau,
            "95ci_low": conf_interval.low,
            "95ci_high": conf_interval.high,
        }

    return stats_dict


def plot_model_preds_scatter(loss_df, lab, fn=None, stats_dict={}):
    # Set so the legend looks nicer
    legend_text_mapper = {-1: "Below Range", 0: "In Range", 1: "Above Range"}
    loss_df["Assay Range"] = list(map(legend_text_mapper.get, loss_df["in_range"]))

    in_range_idx = loss_df["in_range"] == 0
    df_in_range = loss_df.loc[in_range_idx, :]
    fg = sns.relplot(
        data=loss_df,
        x="target",
        y="pred",
        col="split",
        style="Assay Range",
        markers={"Below Range": "<", "In Range": "o", "Above Range": ">"},
        style_order=["Below Range", "In Range", "Above Range"],
    )

    # Figure title
    fg.figure.subplots_adjust(top=0.9)
    fg.figure.suptitle(lab)

    # Axes bounds
    min_val = loss_df.loc[:, ["target", "pred"]].values.flatten().min() - 0.5
    max_val = loss_df.loc[:, ["target", "pred"]].values.flatten().max() + 0.5

    for ax in fg.axes[0]:
        # Set axis labels
        ax.set_ylabel("Predicted pIC50")
        ax.set_xlabel("Experimental pIC50")

        # Make it a square
        ax.set_aspect("equal", "box")
        ax.set_xlim((min_val, max_val))
        ax.set_ylim((min_val, max_val))

        # Plot y=x line
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="black",
            ls="--",
        )

        # Shade 0.5 pIC50 and 1 pIC50 regions
        ax.fill_between(
            [min_val, max_val],
            [min_val - 0.5, max_val - 0.5],
            [min_val + 0.5, max_val + 0.5],
            color="gray",
            alpha=0.2,
        )
        ax.fill_between(
            [min_val, max_val],
            [min_val - 1, max_val - 1],
            [min_val + 1, max_val + 1],
            color="gray",
            alpha=0.2,
        )

    if not stats_dict:
        stats_dict = calculate_statistics(df_in_range)
    for i, sp in enumerate(["train", "val", "test"]):
        mae_str = (
            "MAE: "
            f"${stats_dict[sp]['mae']['value']:0.3f}"
            f"^{{{stats_dict[sp]['mae']['95ci_high']:0.3f}}}"
            f"_{{{stats_dict[sp]['mae']['95ci_low']:0.3f}}}$"
        )
        rmse_str = (
            "RMSE: "
            f"${stats_dict[sp]['rmse']['value']:0.3f}"
            f"^{{{stats_dict[sp]['rmse']['95ci_high']:0.3f}}}"
            f"_{{{stats_dict[sp]['rmse']['95ci_low']:0.3f}}}$"
        )
        sp_r_str = (
            "Spearman's $\\rho$: "
            f"${stats_dict[sp]['sp_r']['value']:0.3f}"
            f"^{{{stats_dict[sp]['sp_r']['95ci_high']:0.3f}}}"
            f"_{{{stats_dict[sp]['sp_r']['95ci_low']:0.3f}}}$"
        )

        tau_str = (
            "Kendall's $\\tau$: "
            f"${stats_dict[sp]['tau']['value']:0.3f}"
            f"^{{{stats_dict[sp]['tau']['95ci_high']:0.3f}}}"
            f"_{{{stats_dict[sp]['tau']['95ci_low']:0.3f}}}$"
        )

        metrics_text = "\n".join([mae_str, rmse_str, sp_r_str, tau_str])
        fg.axes[0, i].text(
            x=0.575,
            y=0.275,
            s=metrics_text,
            transform=fg.axes[0, i].transAxes,
            ha="left",
            va="top",
        )

    if fn:
        plt.savefig(fn, dpi=200, bbox_inches="tight")


def plot_stats_summary(stats_df, fn=None):
    # Reform DF into dict
    stats_dict = {
        lab: (
            g.groupby("split")[["statistic", "value", "95ci_low", "95ci_high"]]
            .apply(lambda x: x.set_index("statistic").to_dict(orient="index"))
            .to_dict()
        )
        for lab, g in stats_df.groupby("lab")
    }

    # Make the initial plots
    fg = sns.catplot(
        stats_df,
        x="statistic",
        y="value",
        order=["mae", "rmse", "sp_r", "tau"],
        hue="lab",
        hue_order=sorted(stats_df["lab"].unique()),
        col="split",
        col_order=["train", "val", "test"],
        kind="bar",
        errorbar=None,
    )

    # Adjust the axes
    for split, ax in zip(["train", "val", "test"], fg.axes[0]):
        # Plot the error bars
        for patch, (lab, statistic) in zip(
            ax.patches,
            product(sorted(stats_df["lab"].unique()), ["mae", "rmse", "sp_r", "tau"]),
        ):
            x = patch.get_x() + 0.5 * patch.get_width()
            y = patch.get_height()
            ax.errorbar(
                x=x,
                y=y,
                yerr=[
                    [
                        stats_dict[lab][split][statistic]["value"]
                        - stats_dict[lab][split][statistic]["95ci_low"]
                    ],
                    [
                        stats_dict[lab][split][statistic]["95ci_high"]
                        - stats_dict[lab][split][statistic]["value"]
                    ],
                ],
                color="black",
            )

        # Fix labels
        ax.set_xlabel("Statistic")
        ax.set_xticklabels(["MAE", "RMSE", "Spearman's $\\rho$", "Kendall's $\\tau$"])

    # Only the first ax has a ylabel
    fg.axes[0, 0].set_ylabel("Statistic Value")

    # Adjust legend title
    fg.legend.set_title("Model")

    if fn:
        plt.savefig(fn, dpi=200, bbox_inches="tight")


def reform_loss_dict(loss_dict, pred_epoch=-1):
    """
    Reform a loss_dict into a DataFrame with the following columns:
    * "compound_id": Compound id
    * "split": Which split this compound was in
    * "target": Target (experimental) value of this compound
    * "in_range": Whether this measurement was below (-1) within (0) or above (1) the
                  dynamic range of the assay
    * "pred": Model prediction at the epoch given by `pred_epoch`

    Parameters
    ----------
    loss_dict : dict
        Dict organized as {split: {compound_id: {exp_dict}}}, where exp_dict contains
        at minimum values for "target", "in_range", and "preds"
    pred_epoch : int, default=-1
        Which epoch to take the prediction value at (defaults to last)

    Returns
    -------
    pandas.DataFrame
        Reformed DataFrame, as described above
    """
    all_split = []
    all_compound_id = []
    all_target = []
    all_range = []
    all_pred = []
    for sp, sp_dict in loss_dict.items():
        for compound_id, cpd_dict in sp_dict.items():
            all_split.append(sp)
            all_compound_id.append(compound_id)
            all_target.append(cpd_dict["target"])
            all_range.append(cpd_dict["in_range"])
            all_pred.append(cpd_dict["preds"][pred_epoch])

    df = pandas.DataFrame(
        {
            "compound_id": all_compound_id,
            "split": all_split,
            "target": all_target,
            "in_range": all_range,
            "pred": all_pred,
        }
    )
    return df
