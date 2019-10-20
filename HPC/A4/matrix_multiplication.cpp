#include <iostream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

void matrixVectorMultiplication(){

    int n;
    cout << "Enter dimension : ";
    cin >> n;

    int results[n] = {0};
    int matrix[n][n];
    cout << "Enter matrix elements" << endl;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            cin >> matrix[i][j];

    int vector[n];
    cout << "\nEnter vector elements" << endl;
    for(int i = 0; i < n; i++)
        cin >> vector[i];

    #pragma omp parallel
    {
        int private_results[n] = {0};
        #pragma omp for
        for(int i = 0; i < n; i++)
            for(int j = 0; j < n; j++)
                private_results[i] += vector[j]*matrix[i][j];
        #pragma omp critical
        {
            for(int i = 0; i < n; i++)
                results[i] += private_results[i];
        }
    }
    cout << "Result : ";
    for(int i = 0; i < n; i++) cout << results[i] << " ";
    cout << endl;
}

void matrixMatrixMultiplication(){
    int n;
    cout << "Enter dimension : ";
    cin >> n;

    int matrix1[n][n];
    cout << "Enter matrix1 elements" << endl;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            cin >> matrix1[i][j];

    int matrix2[n][n];
    cout << "Enter matrix2 elements" << endl;
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            cin >> matrix2[i][j];

    int result[n][n];
    #pragma omp for
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            result[i][j] = 0;
    #pragma omp for
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                result[i][j] += matrix1[i][k]*matrix2[k][j];

    cout << "Result" << endl;
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++)
            cout << result[i][j] << " ";
        cout << endl;
    }
}

int main(){
    cout << "1. Matrix-Vector Multiplication" << endl;
    cout << "2. Matrix-Matrix Multiplication" << endl;
    int choice;
    cout << "Enter choice : ";
    cin >> choice;

    if(choice == 1)
        matrixVectorMultiplication();
    else
        matrixMatrixMultiplication();
}
/*
1. Matrix-Vector Multiplication
2. Matrix-Matrix Multiplication
Enter choice : 1
Enter dimension : 3
Enter matrix elements
1 2 3
4 5 6
7 8 0

Enter vector elements
1 2 3
Result : 14 32 23

1. Matrix-Vector Multiplication
2. Matrix-Matrix Multiplication
Enter choice : 2
Enter dimension : 3
Enter matrix1 elements
1 2 3
4 5 6
7 8 9
Enter matrix2 elements
1 0 0
0 1 0
0 0 1
Result
1 2 3
4 5 6
7 8 9

*/
