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
N_TRIPLET = 1667
N_TWIN = 20000
SINGLE_BLOCK_SIZE = 300
TRIPLET_BLOCK_SIZE = 400
TWIN_BLOCK_SIZE = 200

## Default data source path
INITIAL_SUBMISSION = './twtr.csv'

## Initialize Happiness dictionary
print "INITIALIZE DATA..."
CHILD_PREF = pd.read_csv('../input/child_wishlist_v2.csv', header=None).drop(0, 1).values
GIFT_PREF = pd.read_csv('../input/gift_goodkids_v2.csv', header=None).drop(0, 1).values


'''
==========================
Here we build a dictionary to compute single scores for all three types of children
'''
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

'''
==========================
Here we build a 2D hash_table to compute single scores for all the triplets
'''
print "INITIALIZE TRIPLETS' HAPPINESS..."
TRIPLET_GIFT_HAPPINESS = np.zeros((N_TRIPLET, N_GIFT_TYPE))
TRIPLET_CHILD_HAPPINESS = np.zeros((N_TRIPLET, N_GIFT_TYPE))
pbar.setBar(N_TRIPLET)
for i in range(N_TRIPLET):
    pbar.show(i)
    for g in range(N_GIFT_TYPE):
        TRIPLET_GIFT_HAPPINESS[i][g] = GIFT_HAPPINESS[g][3*i] + GIFT_HAPPINESS[g][3*i + 1] + GIFT_HAPPINESS[g][3*i +2]
        TRIPLET_CHILD_HAPPINESS[i][g] = CHILD_HAPPINESS[3*i][g]+CHILD_HAPPINESS[3*i+1][g]+CHILD_HAPPINESS[3*i+2][g]

'''
==========================
Here we build a 2D hash_table to compute single scores for all the twins
'''
print "INITIALIZE TWINS' HAPPINESS..."
TWIN_GIFT_HAPPINESS = np.zeros((N_TWIN, N_GIFT_TYPE))
TWIN_CHILD_HAPPINESS = np.zeros((N_TWIN, N_GIFT_TYPE))
pbar.setBar(N_TWIN)
for i in range(N_TWIN):
    pbar.show(i)
    for g in range(N_GIFT_TYPE):
        TWIN_GIFT_HAPPINESS[i][g] = GIFT_HAPPINESS[g][TRIPLETS + i*2] + GIFT_HAPPINESS[g][TRIPLETS + i*2 + 1]
        TWIN_CHILD_HAPPINESS[i][g] = CHILD_HAPPINESS[TRIPLETS + i*2][g] + CHILD_HAPPINESS[TRIPLETS + i*2 + 1][g]


'''
==========================
COMPUTE SUMMATION OF HAPPINESS FOR SINGLE CHILDREN
'''
def single_happiness(pred):
    gh = 0.
    ch = 0.
    print("COMPUTE SINGLE HAPPINESS...")
    pbar.setBar(len(pred))
    for i, [c,g] in enumerate(pred):
        pbar.show(i)
        gh += GIFT_HAPPINESS[g][c]
        ch += CHILD_HAPPINESS[c][g]
    print('single child happiness', ch)
    print('single gift happiness', gh)
    return gh, ch
    
'''
==========================
COMPUTE SUMMATION OF HAPPINESS FOR TWINS
'''
def twin_happiness(pred):
    gh = 0.
    ch = 0.
    for i, [c,g] in enumerate(pred):
        gh += TWIN_GIFT_HAPPINESS[c][g]
        ch += TWIN_CHILD_HAPPINESS[c][g]
    print('twin child happiness', ch)
    print('twin gift happiness', gh)
    return gh, ch
    
'''
==========================
COMPUTE SUMMATION OF HAPPINESS FOR TRIPLETS
'''
def tri_happiness(pred):
    gh = 0.
    ch = 0.
    for i, [c,g] in enumerate(pred):
        gh += TRIPLET_GIFT_HAPPINESS[c][g]
        ch += TRIPLET_CHILD_HAPPINESS[c][g]
    print('triplet child happiness', ch)
    print('triplet gift happiness', gh)
    return gh, ch
    

'''
==========================
OPTIMIZATION FOR ALL THREE TYPES OF CHILDREN
''' 
### Define a new entropy term
def entropy(gh, ch, g, c):
    return 3.*gh*g*(g + gh) + g**3 + 3.*ch*c*(c + ch) + c**3

### Optimize the total entropy for the singel children
def optimize_single_block(child_block, gift_block, gh, ch):
    block_size = int(len(child_block))
    C = np.zeros((block_size, block_size))
    for i, c in enumerate(child_block):
        for j, g in enumerate(gift_block):
            C[i, j] = -1.*entropy(gh, ch, GIFT_HAPPINESS[g][c], CHILD_HAPPINESS[c][g])
    row_ind, col_ind = linear_sum_assignment(C)
    return child_block[row_ind], gift_block[col_ind]

### Optimize the total entropy for the twins
def optimize_twin_block(child_block, gift_block, gh, ch):
    block_size = int(len(child_block))
    C = np.zeros((block_size, block_size))
    for i, c in enumerate(child_block):
        for j, g in enumerate(gift_block):
            C[i, j] = -1.*entropy(gh, ch, TWIN_GIFT_HAPPINESS[c][g], TWIN_CHILD_HAPPINESS[c][g])
    row_ind, col_ind = linear_sum_assignment(C)
    return child_block[row_ind], gift_block[col_ind]
    
### Optimize the total entropy for the triplets
def optimize_triplet_block(child_block, gift_block, gh, ch):
    block_size = int(len(child_block))
    C = np.zeros((block_size, block_size))
    for i, c in enumerate(child_block):
        for j, g in enumerate(gift_block):
            C[i, j] = -1.*entropy(gh, ch, TRIPLET_GIFT_HAPPINESS[c][g], TRIPLET_CHILD_HAPPINESS[c][g])
    row_ind, col_ind = linear_sum_assignment(C)
    return child_block[row_ind], gift_block[col_ind]
    
def main_loop():
    ## Restart from the existing progress
    print "RESTARTING..."
    print "READING EXISTING DATAFRAME..."
    subm = pd.read_csv(INITIAL_SUBMISSION)
    
    print "FORMING TRIPLET DATAFRAME..."
    tri_tmp = subm[['ChildId', 'GiftId']][:TRIPLETS].as_matrix()
    tri_mat = np.array([[i, tri_tmp[3*i][1]] for i in range(N_TRIPLET)])
    tri_df = pd.DataFrame({'ChildId' : tri_mat[:,0], 'GiftId' : tri_mat[:, 1]})
    
    print "FORMING TWIN DATAFRAME..."
    twin_tmp = subm[['ChildId', 'GiftId']][TRIPLETS: TWINS].as_matrix()
    twin_mat = np.array([[i, twin_tmp[2*i][1]] for i in range(N_TWIN)])
    twin_df = pd.DataFrame({'ChildId' : twin_mat[:,0], 'GiftId' : twin_mat[:, 1]})
    
    print "EVALUATING INITIAL SCORES..."
    tr_gh, tr_ch = tri_happiness(tri_df[['ChildId', 'GiftId']].values.tolist())
    tw_gh, tw_ch = twin_happiness(twin_df[['ChildId', 'GiftId']].values.tolist())
    si_gh, si_ch = single_happiness(subm[['ChildId', 'GiftId']][TWINS:].values.tolist())
    gh = si_gh + tw_gh + tr_gh
    ch = si_ch + tw_ch + tr_ch
    score = (gh/N_CHILDREN)**3. + (ch/N_CHILDREN)**3.
    print "INITIAL: SCORE = {0}, GIFT HAPPINESS = {1}, CHILD HAPPINESS = {2}".format(
        str(score), str(gh), str(ch))
    
    print "OPTIMIZING SINGLE CHILDREN..."
    # Single Optimization:
    # number of iteration = 20
    single_idx = subm['GiftId'].values
    for step in range(15):
        print "=================  Iteration #{0}  =================".format(str(step))
        perms = np.random.permutation(range(TWINS, N_CHILDREN))
        pbar.setBar(200)
        for j in range(200):
            pbar.show(j)
            child_block = perms[j*SINGLE_BLOCK_SIZE: (j+1)*SINGLE_BLOCK_SIZE]
            gift_block = single_idx[child_block]
            cids, gids = optimize_single_block(child_block, gift_block, gh=gh, ch=ch)
            single_idx[cids] = gids
        subm['GiftId'] = single_idx
        si_gh, si_ch = single_happiness(subm[['ChildId', 'GiftId']][TWINS:].values.tolist())
        gh = si_gh + tw_gh + tr_gh
        ch = si_ch + tw_ch + tr_ch
        score = (gh/N_CHILDREN)**3. + (ch/N_CHILDREN)**3.
        print "SCORE = {0}, GIFT HAPPINESS = {1}, CHILD HAPPINESS = {2}".format(
            str(score), str(gh), str(ch))
    
    
    print "OPTIMIZING TWINS..."
    # Twin Optimization:
    # number of iteration = 4
    twin_idx = twin_df['GiftId'].values
    for step in range(60):
        print "=================  Iteration #{0}  =================".format(str(step))
        perms = np.random.permutation(range(0, N_TWIN))
        for j in range(5):
            child_block = perms[j*TWIN_BLOCK_SIZE: (j+1)*TWIN_BLOCK_SIZE]
            gift_block = twin_idx[child_block]
            cids, gids = optimize_twin_block(child_block, gift_block, gh=gh, ch=ch)
            twin_idx[cids] = gids
        twin_df['GiftId'] = twin_idx
        tw_gh, tw_ch = twin_happiness(twin_df[['ChildId', 'GiftId']].values.tolist())
        gh = si_gh + tw_gh + tr_gh
        ch = si_ch + tw_ch + tr_ch
        score = (gh/N_CHILDREN)**3. + (ch/N_CHILDREN)**3.
        print "SCORE = {0}, GIFT HAPPINESS = {1}, CHILD HAPPINESS = {2}".format(
            str(score), str(gh), str(ch))
            
    print "OPTIMIZING TRIPLETS..."
    # Triplet Optimization:
    # number of iteration = 2
    triplet_idx = tri_df['GiftId'].values
    for step in range(20):
        print "=================  Iteration #{0}  =================".format(str(step))
        perms = np.random.permutation(range(0, N_TRIPLET))
        for j in range(2):
            child_block = perms[j*TRIPLET_BLOCK_SIZE: (j+1)*TRIPLET_BLOCK_SIZE]
            gift_block = triplet_idx[child_block]
            cids, gids = optimize_triplet_block(child_block, gift_block, gh=gh, ch=ch)
            triplet_idx[cids] = gids
        tri_df['GiftId'] = triplet_idx
        tr_gh, tr_ch = tri_happiness(tri_df[['ChildId', 'GiftId']].values.tolist())
        gh = si_gh + tw_gh + tr_gh
        ch = si_ch + tw_ch + tr_ch
        score = (gh/N_CHILDREN)**3. + (ch/N_CHILDREN)**3.
        print "SCORE = {0}, GIFT HAPPINESS = {1}, CHILD HAPPINESS = {2}".format(
            str(score), str(gh), str(ch))
    
    print "FROMING NEW DATAFRAME..."
    triplet_list = []
    for g in tri_df.GiftId.tolist():
        triplet_list += [g]*3
    twin_list = []
    for g in twin_df.GiftId.tolist():
        twin_list += [g]*2
    single_list = subm.GiftId.tolist()[TWINS: ]
    dic = {}
    dic['ChildId'] = [i for i in range(N_CHILDREN)]
    dic['GiftId'] = triplet_list + twin_list + single_list
    output = pd.DataFrame(dic)
    
    print "WRITING BACK TO THE CSV FILE..."
    output[['ChildId', 'GiftId']].to_csv('twtr.csv', index=False)

## Define number of total iterations:
NUM_ITERATION = 2
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
        
    