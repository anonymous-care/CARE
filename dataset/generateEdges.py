# import pandas as pd
import os
import pickle
import gc
import argparse
import random

# roundList2 = ['round24', 'round25']
parser = argparse.ArgumentParser(description="get dataset")
parser.add_argument("--option", help="optional", default="train")
parser.add_argument("--size", help="select_size", default="full")
args = parser.parse_args()
category = args.option
size = float('inf')
roundListAll = ['round24', 'round25', 'round26', 'round27', 'round28',
                'round29',
                'round30', 'round31', 'round32', 'round33', 'round34',
                'round35',
                'round36', 'round37', 'round38']
trainRound = ['round24']
testRound = ['round38']
valRound = ['round31']
trainOutput = "train"
testOutput = "test"
valOutput = "val"
currDirLabel = ''
currDirFollowing = ''
outputDir = ''
partDir = ''
roundList = trainRound
if args.size != "full":
    size = int(args.size)
    outputDir = partDir
if category == "train":
    outputDir += trainOutput
elif category == "test":
    roundList = testRound
    outputDir += testOutput
elif category == "val":
    roundList = valRound
    outputDir += valOutput

print(size)

# data structure
# { userID :[ [{newFriends}, {lostFriends}], [{newFollowers}, {lostFollowers}] ] }
# { userID :{quitTime}}
allUserSet = set()
index = 0
# structure
# {userID:[1,0,.....1]}
# 1 means action, 0 means no action
edges = []
quitUsers = set()
with open(os.path.join(currDirLabel, 'userActiveLabel.pkl'), 'rb') as r:
    ufSet = pickle.load(r)  # all user in set
    keySet = set(ufSet.keys())
    # find all keys
    for key in keySet:
        if ufSet[key] == 0:
            quitUsers.add(int(key))
    r.close()
    gc.collect()
with open(os.path.join(currDirFollowing, 'round24' + 'FriendFollowers.pkl'), 'rb') as r:
    firstRound = pickle.load(r)
    # find all keys
    keySet1 = set(firstRound.keys())
    for key in keySet1:
        allUserSet.add(int(key))
    r.close()
    gc.collect()

print(len(allUserSet))

while index < (len(roundList)):

    print("working on " + roundList[index])
    if category == "train":
        with open(os.path.join(currDirFollowing, roundList[index] + 'FriendFollowers.pkl'), 'rb') as r:
            firstRound = pickle.load(r)
            keySet1 = set(firstRound.keys())
            dicti = {}
            print("...loaded")

            for count, key in enumerate(keySet1):
                if count >= size:
                    break
                if key == 'ERROR':
                    print(f"Skipping key with ERROR: {key}")
                    continue
                userSet = firstRound[key][0]
                for user in userSet:
                    if user == 'ERROR':
                        print(f"Skipping user with ERROR: {user}")
                        continue
                    if int(user) in allUserSet and int(user) not in quitUsers and int(key) not in quitUsers:
                        edges.append((key, user, 5))
            r.close()
            gc.collect()
        index = index + 1
    else:
        roundList1 = []
        roundList2 = []
        if category == "test":
            roundList1 = valRound
            roundList2 = testRound
        elif category == "val":
            roundList1 = trainRound
            roundList2 = valRound
        with open(os.path.join(currDirFollowing, roundList1[index] + 'FriendFollowers.pkl'), 'rb') as r1:
            with open(os.path.join(currDirFollowing, roundList2[index] + 'FriendFollowers.pkl'), 'rb') as r2:
                edges1 = []
                edges2 = []
                firstRound = pickle.load(r1)
                secondRound = pickle.load(r2)
                keySet1 = set(firstRound.keys())
                print("...loaded")

                for count, key in enumerate(keySet1):
                    if count >= size:
                        break
                    if key == 'ERROR':
                        print(f"Skipping key with ERROR: {key}")
                        continue
                    userSet = firstRound[key][0]
                    for user in userSet:
                        if user == 'ERROR':
                            print(f"Skipping user with ERROR: {user}")
                            continue
                        if int(user) in allUserSet and int(user) not in quitUsers and int(key) not in quitUsers:
                            edges1.append((key, user, 5))

                keySet2 = set(secondRound.keys())
                print("...loaded")

                for count, key in enumerate(keySet2):
                    if count >= size:
                        break
                    if key == 'ERROR':
                        print(f"Skipping key with ERROR: {key}")
                        continue
                    userSet = secondRound[key][0]
                    for user in userSet:
                        if user == 'ERROR':
                            print(f"Skipping user with ERROR: {user}")
                            continue
                        if int(user) in allUserSet and int(user) not in quitUsers and int(key) not in quitUsers:
                            edges2.append((key, user, 5))
                edges = list(set(edges2) - set(edges1))
                r1.close()
                r2.close()
                gc.collect()
        index = index + 1
file_name = "edges.txt"
open_file = open(os.path.join(outputDir, file_name), "w")
for u1, u2, r in edges:
    open_file.write(f'{u1} {u2} {r}\n')
open_file.close()
gc.collect()
