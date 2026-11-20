import pandas as pd
import numpy as np

from scipy.stats.contingency import association
from scipy.stats import pointbiserialr

from scr.confiq import TRAIN_DF, TARGET_COLUM

def convert_totalspent_to_num(df: pd.DataFrame, col) -> pd.DataFrame:
    df = df.copy()

    numeric_series = pd.to_numeric(df[col], errors='coerce')

    df[col] = numeric_series.astype(np.float64)

    return df

def type_collect(df, target):

    binary = []
    category = []
    numbers = []

    for col, dtype in df.dtypes.items():
        if col != target:

            if pd.api.types.is_numeric_dtype(dtype):
                if df[col].nunique() == 2:
                    binary.append(col)
                else:
                    numbers.append(col)

            elif pd.api.types.is_string_dtype(dtype):
                if df[col].nunique() == 2:
                    binary.append(col)
                else:
                    category.append(col)

    return category, numbers, binary

def categorical_futures_analiz(df, cat_cols):
    cols_data = {}

    for col in cat_cols:

        nan_indices = df[df[col].isna()].index.tolist()

        values, counts = np.unique(df[col], return_counts=True)

        mode_value = values[np.argmax(counts)]
        mode_count = max(counts)

        cols_data[col] = {"len": len(df[col]),
                          "nan_len": len(nan_indices),
                          "nan_indices": nan_indices,
                          "unique": df[col].nunique(),
                          "moda": mode_value,
                          "mode_count": mode_count,
                          "values":df[col].unique().tolist(),
                          "type_value": df[col].dtype,
                          "value_counts":df[col].value_counts().round(2).tolist(),
                          "value_percent":df[col].value_counts(normalize=True).round(2).tolist()}

    return cols_data

def numeric_futures_analiz(df, num_cols):
    cols_data = {}

    for col in num_cols:

        values = pd.to_numeric(df[col].replace(" ", np.nan), errors='coerce')

        nan_indices = values[values.isna()].index.tolist()

        q1, q3, p95, p99 = np.nanpercentile(values, (25, 75, 95, 99))

        cols_data[col] = {"len": len(values),
                          "nan_len": len(nan_indices),
                          "nan_indices": nan_indices,
                          "type_value":df[col].map(type).unique(),
                          "mean": np.nanmean(values).round(2),
                          "median": np.nanmedian(values).round(2),
                          "min": values.min(),
                          "max": values.max(),
                          "std": values.std().round(2),
                          "Q1": q1.round(2),
                          "Q3": q3.round(2),
                          "p95": p95.round(2),
                          "p99": p99.round(2)}

    return cols_data

def binary_futures_analiz(df, bin_cols):
    cols_data = {}

    for col in bin_cols:

        nan_indices = df[df[col].isna()].index.tolist()

        values, counts = np.unique(df[col], return_counts=True)
        mode_value = values[np.argmax(counts)]
        mode_count = max(counts)

        cols_data[col] = {"len": len(df[col]),
                          "nan_len": len(nan_indices),
                          "nan_indices": nan_indices,
                          "unique": df[col].nunique(),
                          "moda": mode_value,
                          "mode_count": mode_count,
                          "values":df[col].unique().tolist(),
                          "type_value": df[col].map(type).unique(),
                          "value_counts":df[col].value_counts().round(2).tolist(),
                          "value_percent":df[col].value_counts(normalize=True).round(2).tolist()}

    return cols_data

def cramer_v_matrix(df, cat_cols):
    n = len(cat_cols)

    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            col1, col2 = cat_cols[i], cat_cols[j]
            ct = pd.crosstab(df[col1], df[col2])

            if ct.shape[0] < 2 or ct.shape[1] < 2:
                val = 0.0
            else:
                val = association(ct, method="cramer")

            matrix[i, j] = val
            matrix[j, i] = val

    return pd.DataFrame(matrix, index=cat_cols, columns=cat_cols)

def num_correlations(df, num_cols):

    pearson = df[num_cols].corr(method='pearson')
    spearman = df[num_cols].corr(method='spearman')

    return pearson, spearman

def str_binary_correlation(df, bin_cols):
    str_bin = []

    for col in bin_cols:
        if pd.api.types.is_string_dtype(df[col].dtype):
            str_bin.append(col)

    str_bin_corr = cramer_v_matrix(df, str_bin)

    return str_bin_corr

def target_corr(df, target, num_cols, cat_cols, bin_cols):

    results = {}

    results['numbers'] = {}

    for col in num_cols:
        corr, p_value = pointbiserialr(df[target], pd.to_numeric(
            df[col].replace(" ", np.nan), errors='coerce'))

        results['numbers'][col] = {'correlation': round(corr, 3),
                                   'p_value': round(p_value, 4),
                                    'groups': df.groupby(target)[col].agg(['mean', 'median']).round(2).to_dict('index')}

    results['categorical'] = {}

    for col in cat_cols + bin_cols:
        results['categorical'][col] = pd.crosstab(df[col],
                                                  df[target],
                                                  normalize='index').round(3).to_dict('index')

    return results

def explorer(df, target):

    "столбец TotalSpent проблемный. Значения записаны в str хотя фактически там float. Ниже функция конвертации"
    df = convert_totalspent_to_num(df, "TotalSpent")

    "base_info"
    shape_df = df.shape
    class_balance = df["Churn"].value_counts(normalize=True).round(2)

    "futures"
    categorical, numbers, binary = type_collect(df, target)

    "statistics"
    numbers_statistics = numeric_futures_analiz(df, numbers)
    categorical_statistics = categorical_futures_analiz(df, categorical)
    binary_statistics = binary_futures_analiz(df, binary)

    "correlations matrix"
    pearson_n_cor, spearman_s_cor = num_correlations(df, numbers)
    categorical_correlations = cramer_v_matrix(df, categorical)
    binary_correlations = str_binary_correlation(df, binary)

    target_correlation = target_corr(df, target, numbers, categorical, binary)

    "duplication"
    duplicated = df[df.duplicated(keep=False)]

    return {"base_info":{"shape": shape_df,
                        "class_balance": class_balance,
                        "data_category":{"numbers": numbers,
                                          "categorical": categorical,
                                          "binary": binary}},

            "futures":{"categorical": categorical,
                      "numbers": numbers_statistics,
                      "binary": binary_statistics},

            "statistics":{"numbers_stat": numbers_statistics,
                         "categorical_stat": categorical_statistics,
                         "binary_stat": binary_statistics},

            "correlations":{"numbers":{"pearson_number_correlation": pearson_n_cor,
                                       "spearman_number_correlation": spearman_s_cor},
                           "categorical": {"categorical_correlation": categorical_correlations,
                                           "binary_correlation": binary_correlations},
                           "target": {"target_correlation": target_correlation}},

            "duplication":{"duplicate": duplicated}}