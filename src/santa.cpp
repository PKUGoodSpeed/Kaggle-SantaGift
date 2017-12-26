#include<bits/stdc++.h>
#include<cassert>
using namespace std;

typedef vector<int> vi;
typedef vector<double> vd;
typedef pair<int, int> ii;

const int NUM_CHLD = 1000000;
const int NUM_GIFT = 1000;
const int GIFT_LIMT = 1000;
const int WISH_SIZE = 100;
const int PREF_SIZE = 1000;
const int NUM_TRIP = 1667;
const int NUM_TWIN = 20000;

class SantaGifts{
    vector<unordered_map<int, double>> wish_score;
    vector<unordered_map<int, double>> gift_score;
    double getChildWishScore(int childIdx, int giftIdx){
        if(wish_score[childIdx].count(giftIdx)) {
            return wish_score[childIdx][giftIdx];
        }
        return -1./double(WISH_SIZE);
    }
    double getSantaGiftScore(int childIdx, int giftIdx){
        if(gift_score[childIdx].count(giftIdx)){
            return wish_score[childIdx][giftIdx];
        }
        return -1./double(PREF_SIZE);
    }
public:
    SantaGifts(){
        
    }
};

