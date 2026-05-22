import math
import os
import pickle
import gc
import argparse
import numpy as np
from sklearn.linear_model import LinearRegression


# set dir
currDir = ''
outputDir = ''

with open(os.path.join(currDir, 'round29FriendFollowers.pkl'), 'rb') as r1:
    userLabels = {}
    userData = pickle.load(r1)

    keySet = set(userData.keys())
    follower_counts = {}
    for user in keySet:
        follower_set = userData[user][1]
        follower_counts[user] = len(follower_set)

    sorted_users = sorted(follower_counts, key=lambda u: follower_counts[u], reverse=True)

    num_users = len(sorted_users)
    n_top = math.ceil(0.1 * num_users)
    top_users = set(sorted_users[:n_top])

    for user in keySet:
        if user in top_users:
            userLabels[user] = 1
        else:
            userLabels[user] = 0

    file_name = "userPopularityLabel.pkl"
    with open(os.path.join(outputDir, file_name), "wb") as open_file:
        pickle.dump(userLabels, open_file)

    print(f"Labels saved to {file_name}")

    gc.collect()

