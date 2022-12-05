import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ot
import pandas as pd

from faim.util.util import normalizeRowsToOne, scoresByGroup
from faim.visualization.plots import plotScoreHistsPerGroup

DEBUG_SA = 0
DEBUG_SIGMA = 1
SA_COLNAME = "SA_scores"


class FairnessInterpolationAlgorithm:
    """
    interpolates between three mutually exclusive algorithmic fairness definitions namely:
    A) Calibration within groups (or accuracy of prediction)
    B) Balance for the negative class
    C) Balance for the positive class

    The algorithm takes three weights theta_a, theta_b, theta_c as input, which describe the importance
    of the respective fairness definition for the user.

    With these weights, the predicted scores are transformed into a fairer representation, which meets
    criterion A, B and C as per given weight, by use of displacement interpolation per group.

    See https://arxiv.org/abs/2212.00469 for details.
    """

    def __init__(
        self,
        rawData,
        group_names,
        pred_score,
        score_stepsize,
        thetas,
        regForOT,
        path=".",
        plot=False,
    ):
        """
        Arguments:
            rawData {dataframe} -- contains data points as rows and features as columns
            group_names {dict} -- translates from group indicators to group names as strings
            pred_score {string} -- name of column that contains the prediction scores
            score_stepsize {float} -- stepsize between two scores
            thetas {dict} -- keys: group names as int
                             values: vectors of 3 parameters per group,
                             each determining how far a distribution is to be moved towards
                             the barycenter between the three mutually exclusive fairness definitions
                             theta of 1 means that a score distribution is going to match the barycenter
                             theta of 0 means that a score distribution stays exactly where it is
            regForOT {float} -- regularization parameter for optimal transport, see ot docs for details

        Keyword Arguments:
            path {str} -- [description] (default: {'.'})
            plot {bool} -- tells if plots shall be generated (default: {False})
        """

        self.__data = rawData
        self.__predScoreTruncated = pred_score + "_truncated"

        # have some convenience for plots
        self.__groups = group_names
        self.__plotPath = path
        self.__plot = plot

        # calculate bin edges to truncate scores, for histograms and loss matrix size
        self.__binEdges = np.arange(
            rawData[pred_score].min() - score_stepsize,
            rawData[pred_score].max() + score_stepsize,
            score_stepsize,
        )
        self.__numBins = int(len(self.__binEdges) - 1)

        # group predicted scores into bins
        self.__data[self.__predScoreTruncated] = pd.cut(self.__data[pred_score], bins=self.__binEdges, labels=False)

        # normalize data to range in [0, 1]
        x = self.__data[self.__predScoreTruncated]
        self.__data[self.__predScoreTruncated] = (x - min(x)) / (max(x) - min(x))

        y = self.__binEdges
        self.__binEdges = (y - min(y)) / (max(y) - min(y))

        if self.__plot:
            plotScoreHistsPerGroup(
                self.__data,
                self.__binEdges,
                [self.__predScoreTruncated],
                os.path.join(self.__plotPath, "truncatedRawScoreDistributionPerGroup.png"),
                self.__groups,
                xTickLabels=self.__binEdges[:].round(decimals=2),
            )

        # stuff for optimal transport
        # calculate loss matrix
        self.__lossMatrix = ot.utils.dist0(self.__numBins)
        self.__lossMatrix /= self.__lossMatrix.max()

        self.__thetas = thetas
        self.__regForOT = regForOT

    def __plott(
        self, dataframe, filename, xLabel="", yLabel="", xTickLabels=None, yMin=None, yMax=None, isTransportMap=False
    ):
        # FIXME: move this to visualization package?
        mpl.rcParams.update(
            {
                "font.size": 24,
                "lines.linewidth": 3,
                "lines.markersize": 15,
                "font.family": "Times New Roman",
            }
        )
        # avoid type 3 (i.e. bitmap) fonts in figures
        mpl.rcParams["ps.useafm"] = True
        mpl.rcParams["pdf.use14corefonts"] = True
        mpl.rcParams["text.usetex"] = True

        ax = dataframe.plot(kind="bar", use_index=True, legend=False, width=1, figsize=(16, 8))
        if isTransportMap:
            # plot identity line
            identityCol = pd.Series(np.linspace(0, 1, num=len(dataframe.index)))
            identityCol.plot.line(color="grey", linestyle="dashed")
        # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
        #           labels=self.__groups)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        if xTickLabels is not None:
            ax.set_xticklabels(xTickLabels)
        cnt = 0
        for label in ax.xaxis.get_ticklabels():
            if (cnt % 10) != 0:
                label.set_visible(False)
            cnt += 1
        ax.set_ylim(ymin=yMin, ymax=yMax)
        plt.savefig(os.path.join(self.__plotPath, filename), dpi=100, bbox_inches="tight")
        plt.cla()
        plt.close()

    def _getRawScoresByGroupWithCondition(self, conditionCol, conditionVal):
        """
        takes all values from truncated score column (self.__predScoreTruncated) and resorts data such that result contains all scores from
        self.__predScoreTruncated in one column per group.
        Drops all rows where value in conditionCol does not match conditionVal

        Arguments:
            conditionCol {string} -- name of column that contains condition values
            conditionVal {object} -- condition to be met

        Returns:
            [dataframe] -- group labels as column names and scores as column values,
                           columns can contain NaNs if group sizes are not equal
        """

        mask = self.__data[conditionCol] == conditionVal
        return scoresByGroup(self.__data[mask], self.__data["group"].unique(), self.__predScoreTruncated)

    def _dataToHistograms(self, data):
        """
        creates histogram for each column in 'data'
        excludes nans

        Arguments:
            data {pd.DataFrame} -- reference to raw data

        Returns:
            {dataframe} -- columnwise histograms
        """

        def hist(x):
            return np.histogram(x[np.isfinite(x)], bins=self.__binEdges, density=False)[0]

        return pd.DataFrame({cn: hist(data[cn]) for cn in data.columns}, self.__binEdges[:-1])

    def _calculateFairReplacementStrategy(self, group_barycenters, group_histograms_raw):
        """
        Calculates mapping from raw score to fair score using group barycenters.
        This is done by finding the optimal transport matrix, that transports each group's raw score distribution to
        their respective group barycenter.

        Arguments:
            group_barycenters {DataFrame} -- one final barycenter per group that interpolates between the three incompatible
                                             fairness definitions.

        Returns:
            DataFrame -- Optimal transport matrix that translates raw scores into fair scores
                         resulting frame is to be understood as follows: fair score at index 1 replaces
                         raw score at index 1
        """

        fairScoreTranslationPerGroup = pd.DataFrame(columns=self.__groups)
        for groupName in self.__groups:
            # check that vectors are of same length
            if group_histograms_raw[groupName].shape != group_barycenters[groupName].shape:
                raise ValueError("length of raw scores of group and group barycenters should be equal")

            ot_matrix = ot.emd(
                group_histograms_raw[groupName].to_numpy(),
                group_barycenters[groupName].to_numpy(),
                self.__lossMatrix,
            )

            if self.__plot:
                plt.imshow(ot_matrix)
                plt.savefig(
                    os.path.join(self.__plotPath, "OTMatrix_group=" + str(groupName) + ".png"),
                    dpi=100,
                    bbox_inches="tight",
                )
                plt.cla()

            normalized_ot_matrix = normalizeRowsToOne(ot_matrix)

            # this contains a vector per group with len(score_values) entries (e.g. a score range from 1 to 100)
            # results into a group fair score vector of length 100
            fairScoreTranslationPerGroup[groupName] = np.matmul(normalized_ot_matrix, self.__binEdges[:-1].T)

        if self.__plot:
            self.__plott(
                fairScoreTranslationPerGroup,
                "fairScoreReplacementStrategy.png",
                xLabel="raw score",
                yLabel="fair replacement",
                xTickLabels=self.__binEdges[:-1].round(decimals=2),
                yMin=0,
                yMax=1,
                isTransportMap=True,
            )

        return fairScoreTranslationPerGroup

    def _replaceRawByFairScores(self, raw, fairScoreTranslationPerGroup, newScores_colName):
        """
        Replaces raw scores of individuals by their fair representation given in @fairScoreTranslationPerGroup.
        Fair scores are given column-wise, with one column for each group.
        Matchings are identified by their indexes.

        example: for a column, the original score at index 0 in @self.__binEdges will be replaced
                 with the fair score at index 0 in @fairScoreTranslationPerGroup
        """

        def replace(rawData):
            # raw scores of 1 are not translated because translation ranges from [0,1), so we set them to 1
            rawData[newScores_colName] = 1
            rawScores = rawData[self.__predScoreTruncated]
            groupName = rawData["group"].iloc[0]
            fairScores = fairScoreTranslationPerGroup[groupName]
            for index, fairScore in fairScores.iteritems():
                range_left = self.__binEdges[index]
                range_right = self.__binEdges[index + 1]
                replaceAtIndex = (rawScores >= range_left) & (rawScores < range_right)
                rawData.loc[replaceAtIndex, newScores_colName] = fairScore
            return rawData

        raw = raw.groupby(["group"], as_index=False, sort=False).apply(replace)

        fairEdges = sorted(raw[newScores_colName].unique())
        roundedFairEdges = [round(elem, 2) for elem in fairEdges]
        if self.__plot:
            plotScoreHistsPerGroup(
                raw,
                fairEdges,
                [newScores_colName, self.__predScoreTruncated],
                os.path.join(self.__plotPath, newScores_colName + "DistributionPerGroup.png"),
                self.__groups,
                xTickLabels=roundedFairEdges,
            )
            # plot new scores for true positives and true negatives
            mask = raw["groundTruthLabel"] == 1
            plotScoreHistsPerGroup(
                raw[mask],
                fairEdges,
                [newScores_colName],
                os.path.join(
                    self.__plotPath,
                    newScores_colName + "DistributionPerGroup_truePositives.png",
                ),
                self.__groups,
                xTickLabels=roundedFairEdges,
            )
            mask = raw["groundTruthLabel"] == 0
            plotScoreHistsPerGroup(
                raw[mask],
                fairEdges,
                [newScores_colName],
                os.path.join(
                    self.__plotPath,
                    newScores_colName + "DistributionPerGroup_trueNegatives.png",
                ),
                self.__groups,
                xTickLabels=roundedFairEdges,
            )
        return raw

    def _compute_SA_scores(self, df, group_col, truncated_score_col, ground_truth_col, out_col):
        merge_cols = [group_col, truncated_score_col]
        mean_positive_agg = {out_col: (ground_truth_col, np.mean)}
        lambda_plus_all_groups = df.groupby(merge_cols, sort=False).agg(**mean_positive_agg)
        return df.merge(lambda_plus_all_groups, how="left", left_on=merge_cols, right_index=True)

    def _calculate_muA_perGroup(self):
        """
        calculates Eq 2.1 from the paper and the corresponding score distribution μ_A
        S^A (and Eq. 2.1) ensures the following statement:
        the average predicted score of a group should equal its probability of being positive in the ground truth
        """
        self.__data = self._compute_SA_scores(
            self.__data,
            "group",
            self.__predScoreTruncated,
            "groundTruthLabel",
            SA_COLNAME,
        )
        SA_scoresByGroup = scoresByGroup(self.__data, list(self.__groups.keys()), SA_COLNAME)
        muA_perGroup = self._dataToHistograms(SA_scoresByGroup)
        if self.__plot:
            self.__plott(muA_perGroup, "muA_PerGroup.png", xLabel="muA score", yLabel="density")
        # clean up SA_column, it's not needed anymore
        # self.__data = self.__data.drop(columns=[SA_COLNAME])
        return muA_perGroup

    def _calculate_sigmaBar_and_muT(self, conditionCol, conditionVal, colName, plotFilename):
        """
        this function calculates Eq 2.3 in the paper which is the barycenter of the distributions σ^[+-] for each group.
        σ^[+-] is defined in Eq 2.2 and a prerequisite for calculating σ-bar (i.e., Eq. 2.3)

        FIXME: update comment to match these equation numbers with those in paper, when paper is ready
        """

        # FIRST STEP:
        # calculate σ^- (eq 2.2) (for σ^+ replace "negative" by "positive", rest of calculation stays the same):
        # number of people from group t that have score s and are truely negative *
        # number of people from group t that have score s
        # above divided by number of people from group t that are truely negative

        # get scores by group and convert them into a histogram
        groupScoresWithCondition = self._getRawScoresByGroupWithCondition(conditionCol, conditionVal)
        groupScoresWithConditionAsHist = self._dataToHistograms(groupScoresWithCondition)

        if self.__plot:
            plotScoreHistsPerGroup(
                self.__data[self.__data[conditionCol] == conditionVal],
                self.__binEdges,
                [self.__predScoreTruncated],
                os.path.join(
                    self.__plotPath,
                    "predScoresPerGroupWithCondition_" + conditionCol + "=" + str(conditionVal) + "_AsHistograms.png",
                ),
                self.__groups,
                xTickLabels=self.__binEdges[:].round(decimals=2),
            )

        # calculate numerator of Eq. 2.2 (turns out the denominator in λ^[+-]_t and ν_t are canceling each other out
        # and hence, we need only the number of people in group t that have score s and are truely negative)
        numerator = pd.DataFrame(
            groupScoresWithConditionAsHist.values,
            dtype="long",
            columns=groupScoresWithConditionAsHist.columns,
            index=groupScoresWithConditionAsHist.index,
        )

        # calculate group sizes in total and percent
        groupSizes = groupScoresWithCondition.count()
        groupSizesPercent = groupSizes / len(self.__data.loc[self.__data[conditionCol] == conditionVal])

        # finally, calculate σ^[+-]_t (Eq.10):
        # divide previously calculated numerator by number of people per group who are truely negative (independent of s)
        sigmaPerGroup = numerator / groupSizes.values

        if self.__plot:
            self.__plott(
                sigmaPerGroup,
                "sigmaPerGroup_" + conditionCol + "=" + str(conditionVal) + ".png",
                xLabel="normalized score",
                yLabel="Density",
                xTickLabels=self.__binEdges[:-1].round(decimals=2),
            )

        # check if integral over distribution of the two groups is equal (i.e. sum of values per group in histogram should be equal)
        sums = sigmaPerGroup.sum()
        assert math.isclose(sums.values[0], sums.values[1], rel_tol=1e-9)

        # SECOND STEP: calculate σ-bar as barycenter of group sigmas σ^[+-]_t
        # compute general barycenter of all score distributions
        sigmaBar = ot.bregman.barycenter(
            sigmaPerGroup.to_numpy(),
            self.__lossMatrix,
            self.__regForOT,
            weights=groupSizesPercent.values,
            verbose=True,
            log=True,
        )[0]
        print(
            "Sum of barycenter between groups for condition "
            + conditionCol
            + "="
            + str(conditionVal)
            + ": "
            + str(sigmaBar.sum())
        )
        if self.__plot:
            self.__plott(pd.DataFrame(sigmaBar), plotFilename, xLabel="normalized score")

        # create a dataframe with sigmaBar as barycenter for each group
        sigmaBars = pd.DataFrame()
        for group in self.__data["group"].unique():
            sigmaBars[group] = sigmaBar

        s = groupScoresWithConditionAsHist.sum()
        groupScoresWithConditionAsHist /= s
        scoreTranslation = self._calculateFairReplacementStrategy(sigmaBars, groupScoresWithConditionAsHist)

        # do translation only for those who are true negative (resp. true positive)
        # keep old scores for the others
        self.__data[colName] = self.__data[self.__predScoreTruncated]
        replacementIndices = self.__data[conditionCol] == conditionVal
        self.__data.loc[replacementIndices, colName] = self._replaceRawByFairScores(
            self.__data[self.__data[conditionCol] == conditionVal],
            scoreTranslation,
            colName,
        )

        # return the scores as distribution
        newScoresByGroup = scoresByGroup(self.__data, self.__data["group"].unique(), colName)
        muTPerGroup = self._dataToHistograms(newScoresByGroup)
        if self.__plot:
            self.__plott(
                muTPerGroup,
                colName + "DistributionPerGroup.png",
                xLabel=colName + " score",
                yLabel="density",
            )
        return muTPerGroup

    def run(self):

        muA_perGroup = self._calculate_muA_perGroup()
        muB_perGroup = self._calculate_sigmaBar_and_muT("groundTruthLabel", 0, "SB", "sigmaBarMinus.png")
        muC_perGroup = self._calculate_sigmaBar_and_muT("groundTruthLabel", 1, "SC", "sigmaBarPlus.png")

        # each group has a final barycenter, which is calculated as the θ-weighted barycenter between the distributions μ^A_t, μ^B and μ^C
        # hence for each group, we need three θ-values which determine the importance of SA, SB and SC respectively for that group
        # FIXME: parse thetas as dataframe with one theta-vector per group in each column and group names as column names
        groupFinalBarycenters = pd.DataFrame()
        for group in muA_perGroup:
            # normalize each array of group thetas to add up to 1, because they will be used as barycenter weights later
            groupThetas = np.array(self.__thetas.get(group)) / np.array(self.__thetas.get(group)).sum()

            barycenters = pd.DataFrame()
            barycenters["muA"] = muA_perGroup[group]
            barycenters["muB"] = muB_perGroup[group]
            barycenters["muC"] = muC_perGroup[group]

            self.__plott(
                pd.DataFrame(barycenters),
                "muAmuBmuC_group=" + str(group) + ".png",
                xLabel="mu score",
            )

            # all 3 distributions must sum up to the same value
            if barycenters["muA"].sum() != 1:
                s = barycenters["muA"].sum()
                barycenters["muA"] /= s
            if barycenters["muB"].sum() != 1:
                s = barycenters["muB"].sum()
                barycenters["muB"] /= s
            if barycenters["muC"].sum() != 1:
                s = barycenters["muC"].sum()
                barycenters["muC"] /= s
            self.__plott(
                pd.DataFrame(barycenters),
                "muAmuBmuC_normalized_group=" + str(group) + ".png",
                xLabel="mu score",
            )
            groupFinalBarycenters[group] = ot.bregman.barycenter(
                barycenters.to_numpy(),
                self.__lossMatrix,
                self.__regForOT,
                weights=groupThetas,
                verbose=True,
                log=True,
            )[0]
            if self.__plot:
                self.__plott(
                    pd.DataFrame(groupFinalBarycenters[group]),
                    "finalBarycenter_group=" + str(group) + ".png",
                    xTickLabels=self.__binEdges[:-1].round(decimals=2),
                    xLabel="normalized score",
                )

        # plot all barycenters in one image for better comparison
        if self.__plot:
            self.__plott(
                pd.DataFrame(groupFinalBarycenters),
                "finalBarycenters.png",
                xTickLabels=self.__binEdges[:-1].round(decimals=2),
                xLabel="normalized score",
            )

        rawScoresByGroup = scoresByGroup(self.__data, list(self.__groups.keys()), self.__predScoreTruncated)
        rawGroupScoresAsHist = self._dataToHistograms(rawScoresByGroup)
        s = rawGroupScoresAsHist.sum()
        rawGroupScoresAsHist /= s
        fairScoreTranslation = self._calculateFairReplacementStrategy(groupFinalBarycenters, rawGroupScoresAsHist)
        return self._replaceRawByFairScores(self.__data, fairScoreTranslation, "fairScore")
