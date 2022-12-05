"""
Created on Sep 27, 2018

@author: meike.zehlike
"""
import numpy as np
import pandas as pd


def scoresByGroup(data, groups, qual_attr):
    """
    calculates new dataframe with groups as columns and per-group scores as rows

    @param groups:                 all possible groups in data as list of ints.
    @param qual_attr:              name of column that contains the quality attribute (only one possible)

    @return: dataframe with group labels as column names and scores per group as column values
    """
    result = pd.DataFrame(dtype=float)
    # select all rows that belong to one group
    for group in groups:
        scoresPerGroup = data.loc[data["group"] == group]
        resultCol = pd.DataFrame(data=scoresPerGroup[qual_attr].values, columns=[group])
        # needs concat to avoid data loss in case new resultCol is longer than already existing result
        # dataframe
        result = pd.concat([result, resultCol], axis=1)
    # in other words it's doing the following (which is less explicit and not faster):
    # pd.DataFrame({g: data[data["group"] == g].reset_index()[qual_attr] for g in groups})
    return result


def normalizeRowsToOne(mat):
    """
    normalizes rows of `mat` to sum up to 1, ignoring NaNs

    @param mat:                    matrix to normalize

    @return: normalized matrix
    """
    # first calculate sum of the entire integral per row
    norm_vec = np.matmul(mat, np.ones(mat.shape[0]))
    # we have to divide each row entry by this integral, hence creating inverse of norm_vec
    inverse_norm_vec = np.reciprocal(norm_vec)
    # replace nans with zero (in case a row was all 0 in the first place)
    inverse_norm_vec = np.nan_to_num(inverse_norm_vec, copy=False)
    norm_matrix = np.diag(inverse_norm_vec)
    return np.matmul(norm_matrix, mat)
