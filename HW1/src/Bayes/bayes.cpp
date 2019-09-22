//
//  main.cpp
//  Bayes
//
//  Created by 许滨楠 on 2019/9/18.
//  Copyright © 2019 许滨楠. All rights reserved.
//

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using namespace std;

// struct to store p(word)
typedef struct stat {
    int s_times;
    float s_p;
    int h_times;
    float h_p;
} stat;

// word list
map<int, stat> dict;

/*
 * training data processing
 */
void training(int);

/*
 * test data processing and predict
 */
void testing(float);

/*
 * training and testing process controller
 */
int main() {
    int train_data_size = 700;
    float judge_threshold = 0.8;
    training(train_data_size);
    testing(judge_threshold);
    return 0;
}

void training(int train_data_size) {
    ifstream feature, label;
    char feature_file[64], label_file[64];
    sprintf(feature_file, "./train/train-features-%d.txt", train_data_size);
    sprintf(label_file, "./train/train-labels-%d.txt", train_data_size);
    feature.open(feature_file, ios::in);
    label.open(label_file, ios::in);
    
    int no, word, times, last_no = 0, isSpam = 0, spamCnt = 0;
    while (feature >> no >> word >> times) {
        if (last_no != no) {
            label >> isSpam;
            if (isSpam) {
                ++spamCnt;
            }
            last_no = no;
        }
        auto it = dict.find(word);
        if (it == dict.end()) {
            stat temp = {0, 0, 0, 0};
            dict[word] = temp;
        }
        if (isSpam) {
            ++dict[word].s_times;
        } else {
            ++dict[word].h_times;
        }
    }
    for (auto it = dict.begin(); it != dict.end(); ++it) {
        if (it->second.h_times == 0) {
            it->second.h_p = 0.01;
        } else {
            it->second.h_p = (float)it->second.h_times / (train_data_size - spamCnt);
        }
        if (it->second.s_times == 0) {
            it->second.s_p = 0.01;
        } else {
            it->second.s_p = (float)it->second.s_times / spamCnt;
        }
//        cout << it->first << " " << it->second.h_times << " " << it->second.h_p << " " << it->second.s_times << " " << it->second.s_p << endl;
//        cout << it->second.s_p / (it->second.s_p + it->second.h_p) << endl;
    }
    feature.close();
    label.close();
    return;
}

bool cmp(const pair<int, float>& a, const pair<int, float>& b) {
    return a.second > b.second;
}

void testing(float judge_threshold) {
    int test_data_size = 260;
    ifstream feature, label;
    feature.open("./test/test-features.txt", ios::in);
    label.open("./test/test-labels.txt", ios::in);
    int no, word, times, isSpam = 0;
    int right = 0;
    feature >> no >> word >> times;
    for (int i = 1; i <= test_data_size; ++i) {
        label >> isSpam;
        map<int, float> msg_dict;
        while (no == i) {
            auto it = dict.find(word);
            if (it == dict.end()) {
                msg_dict[word] = 0.4;
            } else {
                msg_dict[word] = it->second.s_p / (it->second.s_p + it->second.h_p);
            }
            if (!(feature >> no >> word >> times)) {
                break;
            }
        }
        vector< pair<int, float> > tv(msg_dict.begin(), msg_dict.end());
        sort(tv.begin(), tv.end(), cmp);
//        for (auto it = tv.begin(); it != tv.end(); ++it) {
//            cout << it->first << " " << it->second << endl;
//        }
//        cout << endl << endl;
        float a = 1, b = 1, p;
        for (int i = 0; i < tv.size(); ++i) {
            a *= tv[i].second;
            b *= (1 - tv[i].second);
//            cout << a << " " << b << endl;
        }
        p = a / (a + b);
//        cout << p << endl;
        cout << "Test case [" << setw(3) << i << "] probability: " << setw(10) << setprecision(5) << p * 100 << "%, spam judge: " << setw(3) << (p >= judge_threshold ? "YES" : "NO") << ", actually: " << setw(3) << (isSpam ? "YES" : "NO") << endl;
        if ((p >= judge_threshold && isSpam) || (p < judge_threshold && !isSpam)) {
            ++right;
        }
    }
    cout << "prediction accuracy: " << (float)right / test_data_size * 100 << "%." << endl;
    feature.close();
    label.close();
    return;
}
