from typing import Tuple

import sklearn.metrics as skmetr
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import numpy as np
import os

DECIMAL_PLACES = 3


def evaluate(data: pd.DataFrame, pathToResult: str) -> None:
    # calculate model from true scores (decision boundary was set as mean of true scores) for fair label calc
    x = data["true_score"]
    data["true_score"] = (x - min(x)) / (max(x) - min(x))
    boundary = data["true_score"].mean()
    data["fairscoreLabel"] = 0
    data.loc[data["fairScore"] >= boundary, "fairscoreLabel"] = 1

    # convert group column to category
    data["group"] = data["group"].astype("category")

    # start eval string
    resultString = "\n================================================\nSUMMARY FOR RAW SYNTHETIC SCORES \n================================================\n"
    resultString += evaluateWithGLM_trueLabel(data) + "\n"
    resultString += evaluateWithGLM("predictedLabel", data)
    appendString, rawModelPerformance, rawScoreErrorRates = evaluateModelPerformanceAndErrorRates(
        data, "predictedLabel"
    )
    resultString += appendString

    resultString += "\n\n\n================================================\nSUMMARY FOR FAIR SCORES \n================================================\n"
    resultString += evaluateWithGLM("fairscoreLabel", data)
    appendString, fairModelPerformance, fairScoreErrorRates = evaluateModelPerformanceAndErrorRates(
        data, "fairscoreLabel"
    )
    resultString += appendString

    # calc diffs in model performance
    perfDiffs = pd.DataFrame(
        columns=[
            "raw Acc",
            "raw Prec",
            "raw Recall",
            "fair Acc",
            "fair Prec",
            "fair Recall",
            "raw - fair Acc",
            "raw - fair Prec",
            "raw - fair Recall",
        ]
    )
    perfDiffs["raw Acc"] = rawModelPerformance["Accuracy"]
    perfDiffs["raw Prec"] = rawModelPerformance["Precision"]
    perfDiffs["raw Recall"] = rawModelPerformance["Recall"]
    perfDiffs["fair Acc"] = fairModelPerformance["Accuracy"]
    perfDiffs["fair Prec"] = fairModelPerformance["Precision"]
    perfDiffs["fair Recall"] = fairModelPerformance["Recall"]
    perfDiffs["raw - fair Acc"] = rawModelPerformance["Accuracy"] - fairModelPerformance["Accuracy"]
    perfDiffs["raw - fair Prec"] = rawModelPerformance["Precision"] - fairModelPerformance["Precision"]
    perfDiffs["raw - fair Recall"] = rawModelPerformance["Recall"] - fairModelPerformance["Recall"]

    resultString += "\n\n\n================================================\nPERFORMANCE DIFFERENCES \n================================================\n"
    resultString += perfDiffs.to_string()

    # calc diffs in error rates
    errDiffs = pd.DataFrame(
        columns=[
            "raw FPR",
            "raw FNR",
            "raw MCC",
            "fair FPR",
            "fair FNR",
            "fair MCC",
            "raw - fair FPR",
            "raw - fair FNR",
            "raw - fair MCC",
        ]
    )
    errDiffs["raw FPR"] = rawScoreErrorRates["FPR"]
    errDiffs["raw FNR"] = rawScoreErrorRates["FNR"]
    errDiffs["raw MCC"] = rawScoreErrorRates["MCC"]
    errDiffs["fair FPR"] = fairScoreErrorRates["FPR"]
    errDiffs["fair FNR"] = fairScoreErrorRates["FNR"]
    errDiffs["fair MCC"] = fairScoreErrorRates["MCC"]
    errDiffs["raw - fair FPR"] = rawScoreErrorRates["FPR"] - fairScoreErrorRates["FPR"]
    errDiffs["raw - fair FNR"] = rawScoreErrorRates["FNR"] - fairScoreErrorRates["FNR"]
    errDiffs["raw - fair MCC"] = rawScoreErrorRates["MCC"] - fairScoreErrorRates["MCC"]

    resultString += "\n\n\n================================================\nERROR RATE DIFFERENCES \n================================================\n"
    resultString += errDiffs.to_string()

    # write results to file
    resultFile = os.path.join(os.path.dirname(pathToResult), "eval.txt")
    evalFile = open(resultFile, "w")
    evalFile.write(resultString)
    evalFile.close()


def evaluateModelPerformanceAndErrorRates(data: pd.DataFrame, scoreAttr: str) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    model_performances = pd.DataFrame(columns=["Accuracy", "Precision", "Recall"])
    all_error_rates = pd.DataFrame(columns=["FPR", "FNR", "MCC"])

    # collect model performances
    report = skmetr.classification_report(data["groundTruthLabel"], data[scoreAttr], output_dict=True)
    model_performances.loc["all"] = {
        "Precision": round(report.get("weighted avg").get("precision"), DECIMAL_PLACES),
        "Recall": round(report.get("weighted avg").get("recall"), DECIMAL_PLACES),
        "Accuracy": round(report.get("accuracy"), DECIMAL_PLACES),
    }

    # calc error rates
    confusion_matrix_all = pd.crosstab(data["groundTruthLabel"], data[scoreAttr], margins=True)
    tn_all = confusion_matrix_all.loc[0, 0]
    fp_all = confusion_matrix_all.loc[0, 1]
    fn_all = confusion_matrix_all.loc[1, 0]
    tp_all = confusion_matrix_all.loc[1, 1]
    matthews_correlation_coefficient = (tp_all * tn_all - fp_all * fn_all) / np.sqrt(
        (tp_all + fp_all) * (tp_all + fn_all) * (tn_all + fp_all) * (tn_all + fn_all)
    )
    false_positive_rate = round(fp_all / (fp_all + tn_all), DECIMAL_PLACES)
    false_negative_rate = round(fn_all / (fn_all + tp_all), DECIMAL_PLACES)
    all_error_rates.loc["all"] = {
        "FPR": false_positive_rate,
        "FNR": false_negative_rate,
        "MCC": matthews_correlation_coefficient,
    }

    # print stuff
    result_string = "\nERROR RATES ALL INDIVIDUALS \n====================================\n"
    result_string += skmetr.classification_report(data["groundTruthLabel"], data[scoreAttr]) + "\n\n"
    result_string += str(confusion_matrix_all) + "\n\n"

    for group in sorted(data["group"].unique()):
        group_data = data[data["group"] == group]
        # collect model performances
        report = skmetr.classification_report(group_data["groundTruthLabel"], group_data[scoreAttr], output_dict=True)
        model_performances.loc[group] = {
            "Precision": round(report.get("weighted avg").get("precision"), DECIMAL_PLACES),
            "Recall": round(report.get("weighted avg").get("recall"), DECIMAL_PLACES),
            "Accuracy": round(report.get("accuracy"), DECIMAL_PLACES),
        }

        # calc error rates
        confusion_matrix_group = pd.crosstab(group_data["groundTruthLabel"], group_data[scoreAttr], margins=True)
        tn_group = confusion_matrix_group.loc[0, 0]
        fp_group = confusion_matrix_group.loc[0, 1]
        fn_group = confusion_matrix_group.loc[1, 0]
        tp_group = confusion_matrix_group.loc[1, 1]
        false_positive_rate_group = round(fp_group / (fp_group + tn_group), DECIMAL_PLACES)
        false_negative_rate_group = round(fn_group / (fn_group + tp_group), DECIMAL_PLACES)
        matthews_correlation_coefficient_group = (tp_group * tn_group - fp_group * fn_group) / np.sqrt(
            (tp_group + fp_group) * (tp_group + fn_group) * (tn_group + fp_group) * (tn_group + fn_group)
        )
        all_error_rates.loc[group] = {
            "FPR": false_positive_rate_group,
            "FNR": false_negative_rate_group,
            "MCC": matthews_correlation_coefficient_group,
        }

        # print
        result_string += "ERROR RATES GROUP " + str(group) + " INDIVIDUALS \n====================================\n"
        result_string += skmetr.classification_report(group_data["groundTruthLabel"], group_data[scoreAttr]) + "\n\n"
        result_string += str(confusion_matrix_group) + "\n\n"

    return result_string, model_performances, all_error_rates


def evaluateWithGLM(scoreAttr: str, data: pd.DataFrame) -> str:
    decPlaces = 6

    # train logistic regression model
    model = smf.glm(formula=scoreAttr + " ~ group + groundTruthLabel", data=data, family=sm.families.Binomial())
    result = model.fit()
    resultString = str(result.summary()) + "\n\n"

    # calculate the odds
    asHtml = result.summary().tables[1].as_html()
    resultFrame = pd.read_html(asHtml, header=0, index_col=0)[0]
    intercept = resultFrame.loc["Intercept", "coef"]
    control = np.exp(intercept) / (1 + np.exp(intercept))

    for groupCat in data["group"].unique():
        row = resultFrame.filter(regex=r"group..\." + str(groupCat) + ".", axis=0)
        if row.empty:
            continue
        oddsFactor = np.exp(row["coef"]) / (1 - control + (control * np.exp(row["coef"])))
        resultString += (
            "Group "
            + str(groupCat)
            + "'s chance to have a high score compared to non-protected is "
            + str(round(oddsFactor.values[0], decPlaces))
            + "\n"
        )

    return resultString


def evaluateWithGLM_trueLabel(data: pd.DataFrame) -> str:
    decPlaces = 6

    # train logistic regression model
    model = smf.glm(formula="groundTruthLabel ~ group", data=data, family=sm.families.Binomial())
    result = model.fit()
    resultString = str(result.summary()) + "\n\n"

    # calculate the odds
    asHtml = result.summary().tables[1].as_html()
    resultFrame = pd.read_html(asHtml, header=0, index_col=0)[0]
    intercept = resultFrame.loc["Intercept", "coef"]
    control = np.exp(intercept) / (1 + np.exp(intercept))

    for groupCat in data["group"].unique():
        row = resultFrame.filter(regex=r"group..\." + str(groupCat) + ".", axis=0)
        if row.empty:
            continue
        oddsFactor = np.exp(row["coef"]) / (1 - control + (control * np.exp(row["coef"])))
        resultString += (
            "Group "
            + str(groupCat)
            + "'s chance to have a high score compared to non-protected is "
            + str(round(oddsFactor.values[0], decPlaces))
            + "\n"
        )

    return resultString
