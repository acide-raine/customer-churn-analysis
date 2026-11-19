import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scr.exp.explore import explorer, convert_totalspent_to_num
from scr import confiq

DF = convert_totalspent_to_num(confiq.TRAIN_DF, "TotalSpent")
result = explorer(DF, confiq.TARGET_COLUM)

def distributions(df, col, ax):
    sns.histplot(data=df, x=col, ax=ax)

    ax.set_title(f'{col} распределение')
    ax.set_xlabel(col)
    ax.set_ylabel('Частота')

def num_cor(data_num_cor):

    pearson = data_num_cor["pearson_number_correlation"]
    spearman = data_num_cor["spearman_number_correlation"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        pearson,
        annot=True,
        cmap='coolwarm',
        center=0,
        ax=axes[0]
    )
    axes[0].set_title('Корреляция Пирсона')

    sns.heatmap(
        spearman,
        annot=True,
        cmap='coolwarm',
        center=0,
        ax=axes[1]
    )
    axes[1].set_title('Корреляция Спирмана')

    plt.tight_layout()
    plt.show()

def cat_cor(data_cat_cor):

    categorical_cor = data_cat_cor["categorical_correlation"]
    binary_cor = data_cat_cor["binary_correlation"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(
        categorical_cor,
        annot=True,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        ax=axes[0]
    )
    axes[0].set_title("Корреляция категориальных признаков")

    sns.heatmap(
        binary_cor,
        annot=True,
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        ax=axes[1]
    )
    axes[1].set_title("Корреляция бинарных признаков")

    plt.tight_layout()
    plt.show()

def target_cor(data_target_cor):
    target_data = data_target_cor["target_correlation"]

    numeric = pd.DataFrame({
        col: {"Correlation": values["correlation"]}
        for col, values in target_data["numbers"].items()
    }).T

    categorical = {}
    for col, values in target_data["categorical"].items():
        for category, shares in values.items():
            categorical[f"{col}: {category}"] = shares

    categorical = pd.DataFrame(categorical).T

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    sns.heatmap(
        numeric, annot=True, cmap="coolwarm",
        center=0, ax=axes[0]
    )
    axes[0].set_title("Числовые признаки vs Churn")

    sns.heatmap(
        categorical, annot=True, cmap="coolwarm",
        vmin=0, vmax=1, ax=axes[1]
    )
    axes[1].set_title("Категориальные признаки vs Churn")

    plt.tight_layout()
    plt.show()

def visualizatior(df, result, mode):

    categories = result["base_info"]["data_category"]
    numbers = categories["numbers"]
    categorical = categories["categorical"]
    binary = categories["binary"]

    if mode in ["d_n", "d_c", "all"]:

        if mode == "d_n":
            plot_cols = numbers

        elif mode == "d_c":
            plot_cols = categorical + binary

        else:
            plot_cols = numbers + categorical + binary

        for i in range(0, len(plot_cols), 2):
            cols = plot_cols[i:i+2]

            fig, axes = plt.subplots(
                len(cols), 1,
                figsize=(10, 4 * len(cols))
            )

            if len(cols) == 1:
                axes = [axes]

            for ax, col in zip(axes, cols):
                distributions(df, col, ax)

            plt.tight_layout()
            plt.show()
            plt.close(fig)

    if mode in ["n", "all"]:
        num_cor(result["correlations"]["numbers"])

    if mode in ["c", "all"]:
        cat_cor(result["correlations"]["categorical"])

    if mode in ["t", "all"]:
        target_cor(result["correlations"]["target"])

visualizatior(DF, result, "all")