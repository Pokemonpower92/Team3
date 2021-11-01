import os
import pandas as pd
import numpy as np


if __name__ == "__main__":
    data_dir = os.path.abspath(os.path.dirname(__file__)) + "/../data"
    test_df = pd.read_csv(data_dir+"/test.csv", skipinitialspace=True, encoding = "ISO-8859-1")
    test_sa = pd.read_csv(data_dir+"/test_salaries.csv", skipinitialspace=True, encoding = "ISO-8859-1").to_numpy()
    train_df = pd.read_csv(data_dir+"/train.csv", skipinitialspace=True, encoding = "ISO-8859-1")

    test_df.insert(loc=0, column="Salary", value=test_sa)

    all_data = pd.concat([train_df, test_df])

    all_data.to_csv(data_dir+"/all_data.csv", encoding='utf-8')