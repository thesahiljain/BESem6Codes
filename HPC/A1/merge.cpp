#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
using namespace std;

void merge(int* a, int low, int high){
    int* aux = new int[high-low+1];
    int mid = (low+high)/2;
    int i = low, j = mid+1, k = 0;

    while(i <= mid && j <= high)
        aux[k++] = a[i] < a[j] ? a[i++] : a[j++];
    while(i <= mid)
        aux[k++] = a[i++];
    while(j <= high)
        aux[k++] = a[j++];

    for(int k = 0; k < high-low+1; k++)
        a[low+k] = aux[k];
}

void upMergeSort(int* a, int low, int high){
    if(low >= high) return;
    int mid = (low+high)/2;
    upMergeSort(a, low, mid);
    upMergeSort(a, mid+1, high);
    merge(a, low, high);
}

void pMergeSort(int* a, int low, int high){
    if(low >= high) return;
    int mid = (low+high)/2;
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            pMergeSort(a, low, mid);
        }
        #pragma omp section
        {
            pMergeSort(a, mid+1, high);
        }
    }
    merge(a, low, high);
}

int main()
{
    while(true){
        int n;
        cout << "Enter size : ";
        cin >> n;
        int up[n], p[n];
        for(int i = 0; i < n; i++){
            up[i] = rand()%n;
            p[i] = up[i];
        }

        clock_t up_time = clock();
        upMergeSort(up, 0, n-1);
        up_time = clock()-up_time;
        cout << "Merge sort non-parallel runtime : " << up_time << " ms" << endl;

        clock_t p_time = clock();
        pMergeSort(p, 0, n-1);
        p_time = clock()-p_time;
        cout << "Merge sort parallel runtime : " << p_time << " ms" << endl;
    }
}
/*
Enter size : 10000
Merge sort non-parallel runtime : 2 ms
Merge sort parallel runtime : 16 ms
Enter size : 20000
Merge sort non-parallel runtime : 31 ms
Merge sort parallel runtime : 31 ms
Enter size : 50000
Merge sort non-parallel runtime : 74 ms
Merge sort parallel runtime : 57 ms
Enter size : 100000
Merge sort non-parallel runtime : 145 ms
Merge sort parallel runtime : 121 ms
Enter size : 200000
Merge sort non-parallel runtime : 311 ms
Merge sort parallel runtime : 249 ms
*/
