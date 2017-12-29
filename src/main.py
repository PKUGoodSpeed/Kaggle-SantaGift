## Math Library
import os
import pandas as pd
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, Counter

## Visualization
import matplotlib.pyplot as plt
import seaborn as sns

## Showing progress bar
from progress import ProgressBar
pbar = ProgressBar()

## Define Constants
N_CHILDREN = 1000000
N_GIFT_TYPE = 1000
N_GIFT_QUANTITY = 1000
N_GIFT_PREF = 1000
N_CHILD_PREF = 100
TRIPLETS = 5001
TWINS = 45001

## BLOCK DIM
BLOCK_SIZE = 261
N_BLOCKS = int((N_CHILDREN - TWINS + BLOCK_SIZE - 1) / BLOCK_SIZE)

## Default data source path
INITIAL_SUBMISSION = './twtr.csv'

## Initialize Happiness dictionary
print "INITIALIZE DATA..."
CHILD_PREF = pd.read_csv('../input/child_wishlist_v2.csv', header=None).drop(0, 1).values
GIFT_PREF = pd.read_csv('../input/gift_goodkids_v2.csv', header=None).drop(0, 1).values

print "INITIALIZE GIFT HAPPINESS..."
GIFT_HAPPINESS = {}
pbar.setBar(N_GIFT_TYPE)
for g in range(N_GIFT_TYPE):
    pbar.show(g)
    GIFT_HAPPINESS[g] = defaultdict(lambda: -1. / (2 * N_GIFT_PREF))
    for i, c in enumerate(GIFT_PREF[g]):
        GIFT_HAPPINESS[g][c] = 1. * (N_GIFT_PREF - i) / N_GIFT_PREF

print "INITIALIZE CHILD HAPPINESS..."
CHILD_HAPPINESS = {}
pbar.setBar(N_CHILDREN)
for c in range(N_CHILDREN):
    pbar.show(c)
    CHILD_HAPPINESS[c] = defaultdict(lambda: -1. / (2 * N_CHILD_PREF))
    for i, g in enumerate(CHILD_PREF[c]):
        CHILD_HAPPINESS[c][g] = 1. * (N_CHILD_PREF - i) / N_CHILD_PREF

GIFT_IDS = np.array([[g] * N_GIFT_QUANTITY for g in range(N_GIFT_TYPE)]).flatten()

### Function to compute normalized happiness
def my_avg_normalized_happiness(pred):
    total_child_happiness = 0
    total_gift_happiness = np.zeros(1000)
    print "COMPUTE NORMALIZED HAPPINESS..."
    pbar.setBar(len(pred))
    for i, [c,g] in enumerate(pred):
        pbar.show(i)
        total_child_happiness +=  -CHILD_HAPPINESS[c][g]
        total_gift_happiness[g] += -GIFT_HAPPINESS[g][c]
    nch = total_child_happiness / N_CHILDREN
    ngh = np.mean(total_gift_happiness) / 1000
    print('normalized child happiness', nch)
    print('normalized gift happiness', ngh)
    return nch**3. + ngh**3., ngh*N_CHILDREN, nch*N_CHILDREN
    
### Define a new entropy term
def entropy(gh, ch, g, c):
    return 3.*gh*g*(g + gh) + g**3 + 3.*ch*c*(c + ch) + c**3
### Optimize the total entropy
def optimize_block(child_block, current_gift_ids, gh, ch):
    gift_block = current_gift_ids[child_block]
    C = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for i in range(BLOCK_SIZE):
        c = child_block[i]
        for j in range(BLOCK_SIZE):
            g = GIFT_IDS[gift_block[j]]
            C[i, j] = -1. * entropy(gh, ch, GIFT_HAPPINESS[g][c], CHILD_HAPPINESS[c][g])
    row_ind, col_ind = linear_sum_assignment(C)
    return (child_block[row_ind], gift_block[col_ind])
    
def main_loop():
    ## Restart from the existing progress
    print("RESTARTING...")
    subm = pd.read_csv(INITIAL_SUBMISSION)
    initial_anh, g, c = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
    print(initial_anh, g, c)
    subm['gift_rank'] = subm.groupby('GiftId').rank() - 1
    subm['gift_id'] = subm['GiftId'] * 1000 + subm['gift_rank']
    subm['gift_id'] = subm['gift_id'].astype(np.int32)
    current_gift_ids = subm['gift_id'].values
    
    # Optimization:
    # number of iteration = 20
    for i in range(20):
        print "=================  Iteration #{0}  =================".format(str(i))
        child_blocks = np.split(np.random.permutation(range(TWINS, N_CHILDREN)), N_BLOCKS)
        print "OPTIMIZING..."
        pbar.setBar(200)
        for j in range(200):
            pbar.show(j)
            child_block = child_blocks[j]
            cids, gids = optimize_block(child_block, current_gift_ids=current_gift_ids, gh=g, ch=c)
            current_gift_ids[cids] = gids
        subm['GiftId'] = GIFT_IDS[current_gift_ids]
        anh, g, c = my_avg_normalized_happiness(subm[['ChildId', 'GiftId']].values.tolist())
        print(i, anh, g, c)
    print "WRITING BACK TO THE CSV FILE..."
    subm[['ChildId', 'GiftId']].to_csv('twtr.csv', index=False)

## Define number of total iterations:
NUM_ITERATION = 10
if __name__ == '__main__':
    for step in range(NUM_ITERATION):
        print "$$$$$$$$$$$$$$$$$ STEP #{0} START $$$$$$$$$$$$$$$$$".format(str(step+1))
        print "CPP OPTIMIZATION ON SINGLE CHILDREN..."
        os.system('./apps/twtr ./twtr.csv')
        print "CPP OPTIMIZATION FINISHED"
        
        print "PYTHON OPTIMIZATION ON TWINS AND TRIPLETS..."
        main_loop()
        print "PYTHON OPTIMIZATION FINISHED"
        print "\n"
        
    