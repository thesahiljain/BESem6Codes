#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
using namespace std;

void upBubbleSort(int* a, int n){
    for(int i = 0; i < n; i++){
        int first = i%2;
        for(int j = first; j < n-1; j++)
            if(a[j] > a[j+1])
                swap(a[j], a[j+1]);
    }
}

void pBubbleSort(int* a, int n){
    for(int i = 0; i < n; i++){
        int first = i%2;
        #pragma omp parallel for default(none), shared(a, first)
        for(int j = first; j < n-1; j++)
            if(a[j] > a[j+1])
                swap(a[j], a[j+1]);
    }
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
        upBubbleSort(up, n);
        up_time = clock()-up_time;
        cout << "Bubble sort non-parallel runtime : " << up_time << " ms" << endl;

        clock_t p_time = clock();
        pBubbleSort(p, n);
        p_time = clock()-p_time;
        cout << "Bubble sort parallel runtime : " << p_time << " ms" << endl;
    }
}
/*
Enter size : 10
Bubble sort non-parallel runtime : 0 ms
Bubble sort parallel runtime : 0 ms
Enter size : 20
Bubble sort non-parallel runtime : 0 ms
Bubble sort parallel runtime : 0 ms
Enter size : 50
Bubble sort non-parallel runtime : 0 ms
Bubble sort parallel runtime : 0 ms
Enter size : 100
Bubble sort non-parallel runtime : 0 ms
Bubble sort parallel runtime : 0 ms
Enter size : 200
Bubble sort non-parallel runtime : 1 ms
Bubble sort parallel runtime : 0 ms
Enter size : 500
Bubble sort non-parallel runtime : 4 ms
Bubble sort parallel runtime : 4 ms
Enter size : 1000
Bubble sort non-parallel runtime : 17 ms
Bubble sort parallel runtime : 17 ms
Enter size : 2000
Bubble sort non-parallel runtime : 72 ms
Bubble sort parallel runtime : 69 ms
Enter size : 5000
Bubble sort non-parallel runtime : 461 ms
Bubble sort parallel runtime : 453 ms
Enter size : 10000
Bubble sort non-parallel runtime : 1887 ms
Bubble sort parallel runtime : 1876 ms
*/
