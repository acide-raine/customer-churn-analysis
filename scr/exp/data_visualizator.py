import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scr.exp.explore import explorer, convert_totalspent_to_num
from scr import confiq

DF = convert_totalspent_to_num(confiq.TRAIN_DF, "TotalSpent")
result = explorer(DF, confiq.TARGET_COLUM)

def distributions(df, col):
    plt.figure(figsize=(10, 5))

    sns.histplot(data=df, x=col)

    plt.title(f'{col} распределение')
    plt.xlabel(col)
    plt.ylabel('Частота')

    plt.tight_layout()
    plt.show()

def num_cor(pearson, spearman):

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

def cat_cor(categorical_cor, binary_cor):

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

def target_cor(target_data):

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