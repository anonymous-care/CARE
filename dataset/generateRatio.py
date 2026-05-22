import os
import pickle
import gc
import argparse
import random

parser = argparse.ArgumentParser(description="get dataset")

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
# currDirTweet = './UserTweetChange'
currDirLabel = ''
currDirFollowing = './aggregatedRounds'
outputDir = ''
roundList = trainRound

allUserSet = set()
index = 0

ratio = []
quitUsers = set()
inactiveUsers = set()
with open(os.path.join(currDirLabel, 'userActiveLabel.pkl'), 'rb') as r:
    ufSet = pickle.load(r)
    keySet = set(ufSet.keys())
    # find all keys
    for key in keySet:
        if ufSet[key] == 0:
            quitUsers.add(int(key))
        elif ufSet[key] == 1:
            inactiveUsers.add(int(key))
    r.close()
    gc.collect()
with open(os.path.join(currDirFollowing, 'round27' + 'FriendFollowers.pkl'), 'rb') as r:
    firstRound = pickle.load(r)
    keySet1 = set(firstRound.keys())
    for key in keySet1:
        allUserSet.add(int(key))
    r.close()
    gc.collect()

while index < (len(roundList)):

    print("working on " + roundList[index])
    with open(os.path.join(currDirFollowing, roundList[index] + 'FriendFollowers.pkl'), 'rb') as r:
            firstRound = pickle.load(r)
            keySet1 = set(firstRound.keys())
            dicti = {}
            print("...loaded")

            for count, key in enumerate(keySet1):
                if key == 'ERROR':
                    print(f"Skipping key with ERROR: {key}")
                    continue
                userSet = firstRound[key][1]
                allConnections = 0
                inactiveConnections = 0
                nowratio = 0
                for user in userSet:
                    if user == 'ERROR':
                        print(f"Skipping user with ERROR: {user}")
                        continue
                    if int(user) in allUserSet and int(user) not in quitUsers and int(key) not in quitUsers:
                        allConnections+=1
                        if int(user) in inactiveUsers:
                            inactiveConnections+=1
                if allConnections == 0:
                    nowratio = 0
                else:
                    nowratio = round(inactiveConnections/allConnections, 2)
                ratio.append((key, nowratio))

            r.close()
            gc.collect()
    index = index + 1

file_name = "activeRatio.txt"
open_file = open(os.path.join(outputDir, file_name), "w")
for u1, u2 in ratio:
    open_file.write(f'{u1} {u2}\n')
open_file.close()
gc.collect()
