#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#define MAX 1000
using namespace std;

void serialAddition(){
    ifstream file1("vector1.txt");
    ifstream file2("vector2.txt");
    ofstream file3("result_serial.txt");

    long int x, y;

    for(int i = 0; i < MAX; i++){
        file1 >> x;
        file2 >> y;
        file3 << (x+y) << endl;
    }

    file1.close();
    file2.close();
    file3.close();
}

void parallelAddition(){
    ifstream file1("vector1.txt");
    ifstream file2("vector2.txt");
    ofstream file3("result_parallel.txt");

    long int x, y;
    #pragma omp parallel for
    for(int i = 0; i < MAX; i++){
        file1 >> x;
        file2 >> y;
        file3 << (x+y) << endl;
    }

    file1.close();
    file2.close();
    file3.close();
}

int main(){

    clock_t time;
    // Create file 1
    ofstream file1("vector1.txt");
    for(long int i = 0; i < MAX; i++)
        file1 << rand()%MAX << endl;
    file1.close();

    // Create file 2
    ofstream file2("vector2.txt");
    for(long int i = 0; i < MAX; i++)
        file2 << rand()%MAX << endl;
    file2.close();

    time = clock();
    serialAddition();
    time = float(clock()-time)/CLOCKS_PER_SEC*1000;
    cout << "Serial run-time : " << time  << " ms" << endl;

    time = clock();
    parallelAddition();
    time = float(clock()-time)/CLOCKS_PER_SEC*1000;
    cout << "Parallel run-time : " << time << " ms" << endl;
}
/*
Serial run-time : 44 ms
Parallel run-time : 25 ms
*/
