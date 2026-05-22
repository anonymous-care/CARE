import pandas as pd
import os
import pickle
import gc
import argparse

parser = argparse.ArgumentParser(description="get dataset")
parser.add_argument("--option", help="optional", default="train")
args = parser.parse_args()
category = args.option

roundListAll = ['round24round25', 'round25round26', 'round26round27', 'round27round28', 'round28round29',
                'round29round30', 'round30round31', 'round31round32', 'round32round33', 'round33round34',
                'round34round35', 'round35round36', 'round36round37', 'round37round38', 'round38round39']
trainRound = ['round24round25', 'round25round26', 'round26round27', 'round27round28', 'round28round29',
              'round29round30', 'round30round31', 'round31round32', 'round32round33']
testRound = ['round33round34', 'round34round35', 'round35round36']
valRound = ['round36round37', 'round37round38', 'round38round39']

trainOutput = "train"
testOutput = "test"
valOutput = "val"

currDirTweet = ''
currDirFollowing = ''
outputDir = ''

if category == "train":
    roundList = trainRound
    # outputDir += trainOutput
elif category == "test":
    roundList = testRound
    # outputDir += testOutput
elif category == "val":
    roundList = valRound
    # outputDir += valOutput
else:
    raise ValueError("Invalid category. Choose from 'train', 'test', or 'val'.")

users_action = {}

for index, round_name in enumerate(roundList):
    print(f"Working on {round_name}")

    tweet_path = os.path.join(currDirTweet, round_name + 'Change.pkl')
    follow_path = os.path.join(currDirFollowing, round_name + 'Change.pkl')

    with open(tweet_path, 'rb') as r1, open(follow_path, 'rb') as r2:
        firstRound = pickle.load(r1)
        secondRound = pickle.load(r2)

        keySet = set(firstRound.keys()).union(set(secondRound.keys()))

        for key in keySet:
            user_act = 0
            if key in firstRound and any(len(s) > 0 for s in firstRound[key]):
                user_act += len(firstRound[key][0])
                user_act += len(firstRound[key][1])
            if key in secondRound and len(secondRound[key]) > 0 and any(len(s) > 0 for s in secondRound[key][0]):
                user_act += len(secondRound[key][0][0])
                user_act += len(secondRound[key][0][1])

            if key not in users_action:
                users_action[key] = [0] * index

            users_action[key].append(user_act)

        print(f"Processed {len(keySet)} users in {round_name}")

output_path = os.path.join(outputDir, "userActions.pkl")
with open(output_path, "wb") as open_file:
    pickle.dump(users_action, open_file)

print(f"Data saved to {output_path}")

gc.collect()