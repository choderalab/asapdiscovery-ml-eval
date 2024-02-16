import matplotlib.pyplot as plt
import numpy as np
import pandas
from scipy.stats import bootstrap, kendalltau, spearmanr
import seaborn as sns

color_palette = dict(zip(["train", "val", "test"], sns.color_palette()[:3]))


def plot_model_preds_scatter(loss_df, lab, fn=None):
    # fig, ax = plt.subplots(gridspec_kw={"right": 0.825})

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
        # facet_kws={"legend_out": False},
    )

    # Axes bounds
    min_val = loss_df.loc[:, ["target", "pred"]].values.flatten().min() - 0.5
    max_val = loss_df.loc[:, ["target", "pred"]].values.flatten().max() + 0.5

    for ax in fg.axes[0]:
        # Set title and axis labels
        # ax.set_title(lab)
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

        # Shade  0.5 kcal/mol and 1 kcal/mol regions
        ax.fill_between(
            [min_val, max_val],
            [min_val - 0.5 * np.log(10), max_val - 0.5 * np.log(10)],
            [min_val + 0.5 * np.log(10), max_val + 0.5 * np.log(10)],
            color="gray",
            alpha=0.2,
        )
        ax.fill_between(
            [min_val, max_val],
            [min_val - np.log(10), max_val - np.log(10)],
            [min_val + np.log(10), max_val + np.log(10)],
            color="gray",
            alpha=0.2,
        )

    # fg.legend.set_loc("upper right")
    fg.legend.set_bbox_to_anchor((0.9, 0.8, 0.1, 0.1), transform=fg.figure.transFigure)

    # handles, labels = ax.get_legend_handles_labels()
    # labels = [
    #     "Split",
    #     "train",
    #     "val",
    #     "test",
    #     "Assay Range",
    #     "Below Range",
    #     "In Range",
    #     "Above Range",
    # ]
    # ax.legend(
    #     handles=handles,
    #     labels=labels,
    #     # loc="upper right",
    # )

    # Calculate MAEs
    maes = ["MAE"]
    for sp in ["train", "val", "test"]:
        df_tmp = df_in_range.loc[df_in_range["split"] == sp, ["target", "pred"]]
        mae_str = f"{sp}: ${(df_tmp['target'] - df_tmp['pred']).abs().mean():0.3f}"
        # Calculate bootstrapped confidence intervals
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: np.abs(target - pred).mean(),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        mae_str += f"^{{{conf_interval.high:0.3f}}}_{{{conf_interval.low:0.3f}}}$"
        maes.append(mae_str)
    mae_text = "\n".join(maes)

    # Calculate RMSEs
    rmses = ["RMSE"]
    for sp in ["train", "val", "test"]:
        df_tmp = df_in_range.loc[df_in_range["split"] == sp, ["target", "pred"]]
        rmse_str = (
            f"{sp}: ${np.sqrt((df_tmp['target'] - df_tmp['pred']).pow(2).mean()):0.3f}"
        )
        # Calculate bootstrapped confidence intervals
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: np.sqrt(np.power(target - pred, 2).mean()),
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        rmse_str += f"^{{{conf_interval.high:0.3f}}}_{{{conf_interval.low:0.3f}}}$"
        rmses.append(rmse_str)
    rmse_text = "\n".join(rmses)

    # Calculate Spearman r
    sp_rs = ["Spearman's $\\rho$"]
    for sp in ["train", "val", "test"]:
        df_tmp = df_in_range.loc[df_in_range["split"] == sp, ["target", "pred"]]
        sp_r_str = (
            f"{sp}: ${spearmanr(df_tmp['target'], df_tmp['pred']).statistic:0.3f}"
        )
        # Calculate bootstrapped confidence intervals
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: spearmanr(target, pred).statistic,
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        sp_r_str += f"^{{{conf_interval.high:0.3f}}}_{{{conf_interval.low:0.3f}}}$"
        sp_rs.append(sp_r_str)
    sp_r_text = "\n".join(sp_rs)

    # Calculate Kendall's tau
    taus = ["Kendall's $\\tau$"]
    for sp in ["train", "val", "test"]:
        df_tmp = df_in_range.loc[df_in_range["split"] == sp, ["target", "pred"]]
        tau_str = (
            f"{sp}: ${kendalltau(df_tmp['target'], df_tmp['pred']).statistic:0.3f}"
        )
        # Calculate bootstrapped confidence intervals
        conf_interval = bootstrap(
            (df_tmp["target"], df_tmp["pred"]),
            statistic=lambda target, pred: kendalltau(target, pred).statistic,
            method="basic",
            confidence_level=0.95,
            paired=True,
        ).confidence_interval
        tau_str += f"^{{{conf_interval.high:0.3f}}}_{{{conf_interval.low:0.3f}}}$"
        taus.append(tau_str)
    tau_text = "\n".join(taus)

    metrics_text = "\n\n".join([mae_text, rmse_text, sp_r_text, tau_text])
    ax.text(
        x=0.925,
        y=0.75,
        s=metrics_text,
        transform=fg.figure.transFigure,
        ha="left",
        va="top",
    )

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
