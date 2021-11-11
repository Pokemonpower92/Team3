import matplotlib.pyplot as plt
import math
import os.path
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import seaborn as sbn
sbn.set()


if __name__ == '__main__':
    datafile = os.path.abspath(os.path.dirname(__file__)) + "/../data/all_data.csv"
    image_dir = os.path.abspath(os.path.dirname(__file__)) + "/../images/"
    df = pd.read_csv(datafile, skipinitialspace=True)

    salaries = df['Salary'].to_numpy()
    points = df['PTS'].to_numpy()
    ovld = df['Ovrl'].to_numpy()

    bins = math.ceil((salaries.max() - salaries.min())/5)
    hist = np.histogram(salaries, bins=bins)

    plt.title("Salary Distribution")
    plt.ylabel("Number of Players")
    plt.xlabel("Salary (millions of dollars)")
    plt.step(hist[1][:-1:], hist[0], where="mid", color='g')
    plt.savefig(image_dir+'histogram.png')

    plt.clf()
    plt.title("Salary VS Points")
    plt.ylabel("Salary (millions of dollars)")
    plt.xlabel("Points")
    plt.scatter(points, salaries, color='g', alpha=.5)
    plt.plot(np.unique(points), np.poly1d(np.polyfit(points, salaries, 1))(np.unique(points)))
    plt.savefig(image_dir+'svp.png')

    plt.clf()
    plt.title("Salary VS Overall Draft Pick")
    plt.ylabel("Salary (millions of dollars)")
    plt.xlabel("Pick #")
    plt.scatter(ovld[::-1], salaries, color='g', alpha=.5)
    plt.savefig(image_dir+'svd.png')

