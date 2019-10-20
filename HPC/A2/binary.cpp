#include <iostream>
#include <algorithm>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
using namespace std;

int binarySearch(int* a, int low, int high, int key){
    if(low > high) return -1;
    int mid = (high+low)/2;
    if(a[mid] > key) return binarySearch(a, low, mid-1, key);
    if(a[mid] < key) return binarySearch(a, mid+1, high, key);
    return mid;
}

void upSearch(int* a, int size, int* keys, int n){
    cout << "Positions: ";
    clock_t time = clock();
    for(int i = 0; i < n; i++)
        cout << binarySearch(a, 0, size-1, keys[i]) << " ";
    cout << "\nNon-parallel search time : " << ((float)clock()-time)/CLOCKS_PER_SEC*1000 << " ms" << endl;
}

void pSearch(int* a, int size, int* keys, int n){
    cout << "Positions: ";
    clock_t time = clock();
    #pragma omp parallel for default(none), shared(a, size, keys, n)
    for(int i = 0; i < n; i++)
        cout << binarySearch(a, 0, size-1, keys[i]) << " ";
    cout << "\Parallel search time : " << ((float)clock()-time)/CLOCKS_PER_SEC*1000 << " ms" << endl;
}

int main()
{
    int size;
    cout << "Enter array size : ";
    cin >> size;
    int a[size];
    for(int i = 0; i < size; i++)
        a[i] = rand()%size;
    sort(a, a+size);

    int n;
    cout << "Enter number of keys : ";
    cin >> n;
    int keys[n];
    for(int i = 0; i < n; i++)
        cin >> keys[i];

    upSearch(a, size, keys, n);
    pSearch(a, size, keys, n);
}
/*
Enter array size : 10000
Enter number of keys : 4
100 3000 7000 9990
Positions: -1 3631 -1 9993
Non-parallel search time : 3 ms
Positions: -1 3631 -1 9993 Parallel search time : 2 ms
*/