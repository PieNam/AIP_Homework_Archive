//
//  main.cpp
//  AStar
//
//  Created by 许滨楠 on 2019/9/22.
//  Copyright © 2019 许滨楠. All rights reserved.
//

#include <chrono> // precise timer
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

using namespace std;
using namespace chrono;

#define MAX 999

typedef struct city_path {
    int index;
    int cost;
    vector<int> path;
    bool visited;
    int f;
} city_path;

string cities[20] = {"00 Arad", "01 Bucharest", "02 Craiova", "03 Dobreta", "04 Eforie",
                    "05 Fagaras", "06 Giurgiu", "07 Hirsova", "08 Iasi", "09 Lugoj",
                    "10 Mehadia", "11 Neamt", "12 Oradea", "13 Pitesti", "14 RimnicuVilcea",
                    "15 Sibiu", "16 Timisoara", "17 Urziceni", "18 Vaslui", "19 Zerind"};
int map[20][20];
int h[20][20];

/*
 * get random number for relaxing
 */
int getRandom() {
    srand((int)time(0));
    return rand() % 10;
}

/*
 * initialize the map matrix map[n][n]
 */
void initMap() {
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            if (i == j) {
                map[i][j] = 0;
            } else {
                map[i][j] = MAX;
            }
        }
    }
    map[0][15] = map[15][0] = 140; map[0][16] = map[16][0] = 118; map[0][19] = map[19][0] = 75;
    map[1][5] = map[5][1] = 211; map[1][6] = map[6][1] = 90; map[1][13] = map[13][1] = 101; map[1][17] = map[17][1] = 85;
    map[2][3] = map[3][2] = 120; map[2][13] = map[13][2] = 138; map[2][14] = map[14][2] = 146;
    map[3][10] = map[10][3] = 75;
    map[4][7] = map[7][4] = 86;
    map[5][15] = map[15][5] = 99;
    map[7][17] = map[17][7] = 98;
    map[8][11] = map[11][8] = 87; map[8][18] = map[18][8] = 92;
    map[9][10] = map[10][9] = 70; map[9][16] = map[16][9] = 111;
    map[12][15] = map[15][12] = 151; map[12][19] = map[19][12] = 71;
    map[13][14] = map[14][13] = 97;
    map[14][15] = map[15][14] = 80;
    map[17][18] = map[18][17] = 142;
}

/*
 * initialize the cost function h(n)[n]
 */
void initH() {
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 20; ++j) {
            if (map[i][j] != MAX) {
                h[i][j] = map[i][j];
            } else {
                h[i][j] = MAX;
            }
        }
    }
    h[1][0] = h[0][1] = 366; h[1][1] = h[1][1] = 0; h[1][2] = h[2][1] = 160; h[1][3] = h[3][1] = 242;
    h[1][4] = h[4][1] = 161; h[1][5] = h[5][1] = 178; h[1][6] = h[6][1] = 77; h[1][7] = h[7][1] = 151;
    h[1][8] = h[8][1] = 226; h[1][9] = h[9][1] = 244; h[1][10] = h[10][1] = 241; h[1][11] = h[11][1] = 234;
    h[1][12] = h[12][1] = 380; h[1][13] = h[13][1] = 98; h[1][14] = h[14][1] = 193; h[1][15] = h[15][1] = 253;
    h[1][16] = h[16][1] = 329; h[1][17] = h[17][1] = 80; h[1][18] = h[18][1] = 199; h[1][19] = h[19][1] = 374;
}

/*
 * relax the matrix to get full h(n)[n]
 */
void relax() {
    for (int relax_time = 0; relax_time < 5; ++relax_time) {
        for (int i = 0; i < 20; ++i) {
            for (int j = i+1; j < 20; ++j) {
                int original_cost = h[i][j], t_cost = MAX;
                for (int k = 0; k < 20; ++k) {
                    if (h[i][k] == MAX || h[k][j] == MAX) {
                        continue;
                    }
                    int temp = h[i][k] + h[k][j] - getRandom();
                    t_cost = t_cost < temp ? t_cost : temp;
                }
                if (t_cost < original_cost) {
                    h[i][j] = h[j][i] = t_cost;
                }
            }
        }
    }
}

/*
 * dijkstra to get result
 */
city_path dijkstra(int src, int dst) {
    city_path dis[20];
    for (int i = 0; i < 20; ++i) {
        dis[i].index = i;
        dis[i].cost = map[src][i];
        dis[i].visited = false;
        if (map[src][i] != MAX) {
            dis[i].path.push_back(src);
        }
    }
    dis[src].visited = true;
    
    while (true) {
        int min = MAX, index = 0;
        for (int i = 0; i < 20; ++i) {
            if (dis[i].cost < min && !dis[i].visited) {
                min = dis[i].cost;
                index = dis[i].index;
            }
        }
        
        dis[index].visited = true;
        if (index == dst) {
            return dis[index];
        }
        for (int j = 0; j < 20; ++j) {
            if (map[index][j] + dis[index].cost < dis[j].cost && !dis[j].visited) {
                dis[j].cost = map[index][j] + dis[index].cost;
                dis[j].path.clear();
                for (auto it = dis[index].path.begin(); it != dis[index].path.end(); ++it) {
                    dis[j].path.push_back(*it);
                }
                dis[j].path.push_back(index);
            }
        }
    }
}

/*
 * a-star to get result
 */
city_path astar(int src, int dst) {
    city_path dis[20];
    for (int i = 0; i < 20; ++i) {
        dis[i].index = i;
        dis[i].cost = map[src][i];
        dis[i].f = map[src][i] + h[i][dst];
        dis[i].visited = false;
        if (map[src][i] != MAX) {
            dis[i].path.push_back(src);
        }
    }
    dis[src].visited = true;
    
    while (true) {
        int min = MAX, index = 0;
        for (int i = 0; i < 20; ++i) {
            if (dis[i].f < min && !dis[i].visited) {
                min = dis[i].f;
                index = dis[i].index;
            }
        }
        
        dis[index].visited = true;
        if (index == dst) {
            return dis[index];
        }
        for (int j = 0; j < 20; ++j) {
            if (map[index][j] + dis[index].cost < dis[j].cost && !dis[j].visited) {
                dis[j].cost = map[index][j] + dis[index].cost;
                dis[j].f = dis[j].cost + h[j][dst];
                dis[j].path.clear();
                for (auto it = dis[index].path.begin(); it != dis[index].path.end(); ++it) {
                    dis[j].path.push_back(*it);
                }
                dis[j].path.push_back(index);
            }
        }
    }
}

/*
 * display the path of the search result
 */
void displayPath(city_path res) {
    int count = 0;
    cout << "Path: ";
    for (auto it = res.path.begin(); it != res.path.end(); ++it) {
        ++count;
        if (count == 4) {
            cout << endl << "      ";
            count = 1;
        }
        cout << cities[*it] << " -> ";
    }
    cout << cities[res.index] << endl;
}

/*
 * matrix display for debug
 */
void display(int option) {
    if (!option) {
        cout << "========== map  ==========" << endl;
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                cout << setw(3) << map[i][j] << " ";
            }
            cout << endl;
        }
    } else {
        cout << "========== h(n) ==========" << endl;
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                cout << setw(3) <<  h[i][j] << " ";
            }
            cout << endl;
        }
    }
    cout << endl;
}

int main() {
    initMap();
    initH();
    relax();
    // display(0);
    // display(1);
    
    
    string src_s, dst_s;
    int src = 0, dst = 0;
    cout << "Please enter the two cities to perform pathfinding" << endl;
    cout << "Hint: select cities by [NO. like 00, 01, ..., 12]" << endl;
    cout << "                 or by [Name like Arad, ..., Zerind]" << endl;
    cout << "          or simply by [Initial like A/a, ..., Z/z]" << endl;
    cout << "====================================================" << endl;
    cout << "From: "; cin >> src_s;
    cout << "To  : "; cin >> dst_s;
    for (int i = 0; i < 20; ++i) {
        if ((cities[i][0] == src_s[0] && cities[i][1] == src_s[1])
            || (cities[i][3] == (char)toupper(src_s[0]))) {
            src = i;
            break;
        }
    }
    for (int i = 0; i < 20; ++i) {
        if ((cities[i][0] == dst_s[0] && cities[i][1] == dst_s[1])
            || (cities[i][3] == (char)toupper(dst_s[0]))) {
            dst = i;
            break;
        }
    }
    // cout << src << " " << dst << endl;
    
    
    
    auto start = system_clock::now();
    city_path d_res = dijkstra(src, dst);
    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    cout << "====================== Result ======================" << endl;
    cout << "[Dijkstra]" << endl;
    displayPath(d_res);
    cout << "Cost: " << d_res.cost << endl;
    cout << "Algorithm Time Cost: " << double(duration.count()) * microseconds::period::num / microseconds::period::den << "s" << endl;

    cout << endl;
    
    start = system_clock::now();
    city_path a_res = astar(src, dst);
    end = system_clock::now();
    duration = duration_cast<microseconds>(end - start);

    cout << "[A-Star]" << endl;
    displayPath(a_res);
    cout << "Cost: " << a_res.cost << endl;
    cout << "Algorithm Time Cost: " << double(duration.count()) * microseconds::period::num / microseconds::period::den << "s" << endl;
    
    return 0;
}
