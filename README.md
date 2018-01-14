# README
### Website:
https://www.kaggle.com/c/santa-gift-matching#description

### Description:
‘Tis the night before Christmas 
year: two thousand seventeen.
Santa’s grown grouchy, 
borderline mean.
What used to be simple for Old St. Nick, 
is now too puzzling, it’s making him sick!
See, Santa always knew, deep down in his gut, 
what toy each kid wanted–no ifs, ands, or buts.
But fierce population growth, more twins, and toy innovation, 
has left too complex a problem, in dire need of optimization.
“Don’t worry, Mr. Santa”, said an Elf named McMaggle, 
“I have a solution! Have you heard of Kaggle?”
As she explained Kaggle in-depth, Santa’s doubt began turning, 
he became a believer in the magic of...machine learning.
So, Santa’s team needs YOU more than ever this year, 
to solve this painful problem and save Christmas cheer.

### The Challenge:
In this playground competition, you’re challenged to build a toy matching algorithm that maximizes happiness by pairing kids with toys they want. In the dataset, each kid has 10 preferences for their gift (from 1000) and Santa has 1000 preferred kids for every gift available. What makes this extra difficult is that 0.4% of the kids are twins, and by their parents’ request, require the same gift.

### Evaluation:
Your goal is to maximize the 

    Average Normalized Happiness (ANH) = (AverageNormalizedChildHappiness (ANCH) ) ^ 3 + (AverageNormalizedSantaHappiness (ANSH) ) ^ 3

where **NormalizedChildHappiness** is the happiness of each child, divided by the maximum possible happiness, and **NormalizedSantaHappiness** is the happiness of each gift, divided by the maximum possible happiness. 
**Note the cubic terms with ANCH and ANSH.**
in the equation form:

        $$ANCH = \frac{1}{nc}\sum_{i=0}^{n_c-1} \frac{\text{ChildHappiness}}{\text{MaxChildHappiness}}$$
        $$ANCH = \frac{1}{ng}\sum_{i=0}^{ng-1}\frac{\text{GiftHappiness}}{\text{MaxChildHappiness}}$$

$$nc$$ is the number of children. $$ng$$ is the number of gifts

    MaxChildHappiness = len(ChildWishList) * 2,
    MaxGiftHappiness = len(GiftGoodKidsList) * 2.
    ChildHappiness = 2 * GiftOrder if the gift is found in the wish list of the child. 
    ChildHappiness = -1 if the gift is out of the child's wish list. 

Similarly, `GiftHappiness = 2 * ChildOrder` if the child is found in the good kids list of the gift.

    GiftHappiness = -1 if the child is out of the gift's good kids list. 

For example, if a child has a preference of gifts `[5,2,3,1,4]`, and is given gift `3`, then 

    ChildHappiness = [len(WishList)-indexOf(gift_3)] * 2 = [5 - 2] * 2 = 6

If this child is given gift 4, then `ChildHappiness = [5-4] * 2 = 2`
Code sample of `Average Normalized Happiness` can be seen from this [Kernel](https://www.kaggle.com/wendykan/average-normalized-happiness-demo). 

### Submission File:
For each child in the dataset, you will match it with a gift. Remember, the first 0.5% of rows (`ChildId` 0 to 5000) are triplets, and the following 4% (`ChildId` 5001-45000) are twins. 

    ChildId,GiftId
    0,669
    1,669
    2,669
    3,8
    4,8
    5,8
    6,689
    7,689
    8,689

### Src Codes in github:
1. [naive_simulated_annealing.cpp](https://github.com/PKUGoodSpeed/Kaggle-SantaGift/blob/master/src/naive_simulated_annealing.cpp): using naive simulated annealing (swap based)
2. [twtr_simulated_annealing.cpp](https://github.com/PKUGoodSpeed/Kaggle-SantaGift/blob/master/src/twtr_simulated_annealing.cpp): using simulated annealing to deal with twins and triplets
3. [lsa_optimizer.py](https://github.com/PKUGoodSpeed/Kaggle-SantaGift/blob/master/src/lsa_optimizer.py): using scipy.linear_sum_assignment
4. [multi_thread_lsa.py](https://github.com/PKUGoodSpeed/Kaggle-SantaGift/blob/master/src/multi_thread_lsa.py): using scipy.linear_sum_assignment with multithreading.Pool
5. [optimizer_using_openmp.c](https://github.com/PKUGoodSpeed/Kaggle-SantaGift/blob/master/src/optimizer_using_openmp.c): heuristic optimizer by ZFTurbo

