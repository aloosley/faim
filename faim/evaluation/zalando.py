import datetime as dt
import os
from collections import defaultdict
from functools import wraps

import numpy as np
import pandas as pd
import sklearn.metrics as skmetr


def evaluate(data, pathToResult, filename="eval.txt"):
    # set predicted label
    threshold = 0.5
    data["origPredLabel"] = data["pred_score_truncated"] > threshold
    data["fairLabel"] = data["fairScore"] > threshold

    # calculate performance metrics
    origResult = performanceAndErrorRates(data, "origPredLabel")
    fairResult = performanceAndErrorRates(data, "fairLabel")

    # write results into table
    resultString = createLatexTableSnippet(origResult, fairResult)
    resultFile = os.path.join(os.path.dirname(pathToResult), filename)
    evalFile = open(resultFile, "w")
    evalFile.write(resultString)
    evalFile.close()

    # plot rankings with and without FAIM
    plotRankings(data, pathToResult)
    return


def createLatexTableSnippet(origResult, fairResult):
    groupStrings = origResult["group"]
    origResult = origResult.drop(["group"], axis=1)
    fairResult = fairResult.drop(["group"], axis=1)
    deltas = (origResult - fairResult) * -1
    deltas["group"] = groupStrings
    origResult["group"] = groupStrings

    tblOrig = "\\toprule \n \
                Dataset & Parameters & \multicolumn{3}{c}{Performance} & \multicolumn{2}{c}{Error Rates}   \\\ \n \
                \cline{3-5} \cline{6-7} \n \
                &&  Accur. ($\Delta$)  & Precision ($\Delta$) &  Recall ($\Delta$)	& FPR ($\Delta$) & FNR ($\Delta$) \\\    \midrule \n"
    tblFair = ""
    for idx, row in origResult.iterrows():
        deltaRow = deltas.iloc[[idx]].squeeze()
        fairRow = fairResult.iloc[[idx]].squeeze()
        # put correct textcolor per metric delta
        deltaStringWithColors = pd.Series(dtype="string")
        for metricName in ["accuracy", "precision", "recall"]:
            if deltaRow.at[metricName] < 0:
                deltaStringWithColors[metricName] = f"\\textcolor{{red}}{{{deltaRow.at[metricName]:.3f}}}"
            else:
                deltaStringWithColors[metricName] = f"\\textcolor{{forestgreen}}{{{deltaRow.at[metricName]:.3f}}}"
        for metricName in ["FPR", "FNR"]:
            if deltaRow.at[metricName] > 0:
                deltaStringWithColors[metricName] = f"\\textcolor{{red}}{{{deltaRow.at[metricName]:.3f}}}"
            else:
                deltaStringWithColors[metricName] = f"\\textcolor{{forestgreen}}{{{deltaRow.at[metricName]:.3f}}}"

        if row.group == "all":
            tblOrig += f"Zalando & before \\methodname & {row.accuracy:.3f} & {row.precision:.3f} & {row.recall:.3f} & {row.FPR:.3f} & {row.FNR:.3f} \\\ \n"
            tblFair += f"Zalando & $\\theta^X = 1$ & {fairRow.accuracy:.3f} ({deltaStringWithColors.accuracy}) & \
                                    {fairRow.precision:.3f} ({deltaStringWithColors.precision}) & \
                                    {fairRow.recall:.3f} ({deltaStringWithColors.recall}) & \
                                    {fairRow.FPR:.3f} ({deltaStringWithColors.FPR}) & \
                                    {fairRow.FNR:.3f} ({deltaStringWithColors.FNR})\\\ \n"
        elif row.group == "high":
            tblOrig += f"$\;\;\;$ high & & {row.accuracy:.3f} & {row.precision:.3f} & {row.recall:.3f} & {row.FPR:.3f} & {row.FNR:.3f} \\\ \n"
            tblFair += f"$\;\;\;$ high & & {fairRow.accuracy:.3f} ({deltaStringWithColors.accuracy}) & \
                                    {fairRow.precision:.3f} ({deltaStringWithColors.precision}) & \
                                    {fairRow.recall:.3f} ({deltaStringWithColors.recall}) & \
                                    {fairRow.FPR:.3f} ({deltaStringWithColors.FPR}) & \
                                    {fairRow.FNR:.3f} ({deltaStringWithColors.FNR})\\\ \n"
        elif row.group == "low":
            tblOrig += f"$\;\;\;$ low & & {row.accuracy:.3f} & {row.precision:.3f} & {row.recall:.3f} & {row.FPR:.3f} & {row.FNR:.3f} \\\ \n"
            tblFair += f"$\;\;\;$ low && {fairRow.accuracy:.3f} ({deltaStringWithColors.accuracy}) & \
                                    {fairRow.precision:.3f} ({deltaStringWithColors.precision}) & \
                                    {fairRow.recall:.3f} ({deltaStringWithColors.recall}) & \
                                    {fairRow.FPR:.3f} ({deltaStringWithColors.FPR}) & \
                                    {fairRow.FNR:.3f} ({deltaStringWithColors.FNR})\\\ \n"
    tblOrig += "\midrule \n"
    tblFair += "\midrule \n"

    return tblOrig + tblFair


def performanceAndErrorRates(df: pd.DataFrame, predictedLabel):
    eres = defaultdict(list)
    masks = [np.ones(len(df), dtype=bool), df.group == 0, df.group == 1]
    for mask, name in zip(masks, ["all", "high", "low"]):
        y_true = df.loc[mask, "groundTruthLabel"].to_numpy()
        y_pred = df.loc[mask, predictedLabel].to_numpy()
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        # Thus in binary classification, the count of true negatives is C_{0,0}, false positives is
        # C_{0,1}, false negatives is C_{1,0} and true positives is C_{1,1}.
        tn, fp, fn, tp = skmetr.confusion_matrix(y_true, y_pred).ravel()
        metrics = {
            "group": name,
            "accuracy": skmetr.accuracy_score(y_true, y_pred),
            "precision": skmetr.precision_score(y_true, y_pred, average="binary"),
            "recall": skmetr.recall_score(y_true, y_pred, average="binary"),
            # https://en.wikipedia.org/wiki/Confusion_matrix
            "FPR": fp / (fp + tn),
            "FNR": fn / (fn + tp),
        }
        for metric, value in metrics.items():
            eres[metric].append(value)
    return pd.DataFrame(eres)


def plotRankings(data, pathToResult):
    try:
        import plotly.express as px
    except ImportError as err:
        raise ImportError("plotly must be installed to run this function")

    data.loc[data["group"] == 0, "group"] = "high"
    data.loc[data["group"] == 1, "group"] = "low"

    def log_step(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tic = dt.datetime.now()
            result = func(*args, **kwargs)
            time_taken = str(dt.datetime.now() - tic)
            print(f"ran step {func.__name__.ljust(30)}shape={str(result.shape).ljust(16)}{time_taken}s")
            return result

        return wrapper

    @log_step
    def start_pipeline(dataf):
        return dataf.copy()

    # synthetic ranking to show ideal situation
    pos_bin = 30 * [("A", 1)] + 20 * [("B", 1)]
    neg_bin = 40 * [("A", 0)] + 10 * [("B", 0)]

    syn = pd.DataFrame(
        [t for i in range(10) for t in pos_bin] + [t for i in range(20) for t in neg_bin],
        columns=["group", "groundTruthLabel"],
    )
    syn["position"] = np.arange(len(syn))
    syn["bin"] = pd.qcut(syn.position, 30)
    syn["bin_num"] = syn.bin.cat.codes
    per_bin = (
        syn.groupby(["bin_num", "group", "groundTruthLabel"]).size().reset_index().rename(columns={0: "num_items"})
    )
    f = px.bar(
        per_bin,
        x="bin_num",
        y="num_items",
        color="group",
        pattern_shape="groundTruthLabel",
        height=500,
        width=1600,
        pattern_shape_map={0: "", 1: "/"},
        labels={"bin_num": "Bin number", "num_items": "#items"},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    f.update_layout(font={"size": 30, "family": "Times New Roman"}, legend_title_text="")
    f.write_image(os.path.join(os.path.dirname(pathToResult), "synthetic_ranking.png"))

    @log_step
    def add_position(dataf: pd.DataFrame, score_name: str) -> pd.DataFrame:
        dataf[f"position_{score_name}"] = len(dataf) - np.argsort(dataf[score_name].to_numpy()).argsort() - 1
        return dataf

    @log_step
    def add_bin_num(dataf: pd.DataFrame, pos_name: str, num_bins: int) -> pd.DataFrame:
        return dataf.assign(bin_num=lambda syn: pd.qcut(syn[pos_name], num_bins).cat.codes)

    score_name = "pred_score_truncated"
    rf = data.pipe(start_pipeline).pipe(add_position, score_name).pipe(add_bin_num, f"position_{score_name}", 30)
    rf.head()

    per_bin = rf.groupby(["bin_num", "group", "groundTruthLabel"]).size().reset_index().rename(columns={0: "num_items"})
    f = px.bar(
        per_bin,
        x="bin_num",
        y="num_items",
        color="group",
        pattern_shape="groundTruthLabel",
        height=500,
        width=1600,
        pattern_shape_map={0: "", 1: "/"},
        labels={"bin_num": "Bin number", "num_items": "#items"},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    f.update_layout(font={"size": 30, "family": "Times New Roman"}, legend_title_text="")
    f.write_image(os.path.join(os.path.dirname(pathToResult), "real_ranking.png"))

    score_name = "fairScore"
    rf = data.pipe(start_pipeline).pipe(add_position, score_name).pipe(add_bin_num, f"position_{score_name}", 30)
    per_bin = rf.groupby(["bin_num", "group", "groundTruthLabel"]).size().reset_index().rename(columns={0: "num_items"})
    f = px.bar(
        per_bin,
        x="bin_num",
        y="num_items",
        color="group",
        pattern_shape="groundTruthLabel",
        height=500,
        width=1600,
        pattern_shape_map={0: "", 1: "/"},
        labels={"bin_num": "Bin number", "num_items": "#items"},
        color_discrete_sequence=px.colors.qualitative.T10,
    )
    f.update_layout(font={"size": 30, "family": "Times New Roman"}, legend_title_text="")
    f.write_image(os.path.join(os.path.dirname(pathToResult), "fairer_ranking.png"))
    return
