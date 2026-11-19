import pandas as pd

TRAIN_DF = pd.read_csv("/Users/pasha/Projects/churn/CHURN_DATA/train.csv")
TEST_DF = pd.read_csv("/Users/pasha/Projects/churn/CHURN_DATA/test.csv")

TARGET_COLUM = "Churn"