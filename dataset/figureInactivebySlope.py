import math
import os
import pickle
import gc
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression

# figure by using slopes (currently use)
parser = argparse.ArgumentParser(description="get label")
parser.add_argument("--ratio", help="optional", default=1)
args = parser.parse_args()
ratio = float(args.ratio)
# set dir
currDir = ''
outputDir = ''

with open(os.path.join(outputDir, 'userActions.pkl'), 'rb') as r1:
    userLabels = {}
    userData = pickle.load(r1)

    keySet = set(userData.keys())

    all_means = [np.mean(userData[u][:6]) for u in userData]
    global_mean = np.mean(all_means)
    print(global_mean)

    for key in keySet:
        total_rounds = len(userData[key])
        w = 6  # training round size

        if np.all(np.array(userData[key][:6]) == 0):
            userLabels[key] = 0
            continue

        if total_rounds < w:
            userLabels[key] = 2
            continue

        slopes = np.diff(userData[key])
        moving_avg_slopes = np.convolve(slopes, np.ones(w)/w, mode='valid')

        # negative ratios (not used)
        neg_slopes = moving_avg_slopes < 0
        negative_slope_ratio = np.sum(neg_slopes) / len(moving_avg_slopes)

        # linear regression (currently use)
        t = np.arange(len(moving_avg_slopes)).reshape(-1, 1)
        model = LinearRegression().fit(t, moving_avg_slopes.reshape(-1, 1))
        trend_w = model.coef_[0][0]  # slope
        threshold = np.std(moving_avg_slopes)  # sd as dynamic threshold

        # mean (not used)
        mean_all = np.mean(userData[key][:6])
        new_threshold = -(threshold + np.log(mean_all + 1))

        if trend_w < new_threshold:
            userLabels[key] = 1  # Inactive user
        else:
            userLabels[key] = 2  # Active user

    file_name = "userActiveLabel.pkl"
    with open(os.path.join(outputDir, file_name), "wb") as open_file:
        pickle.dump(userLabels, open_file)

    print(f"Labels saved to {file_name}")

    gc.collect()
