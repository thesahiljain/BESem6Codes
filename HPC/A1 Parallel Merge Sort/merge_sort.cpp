#include <iostream>
#include <stdlib.h>
#include <omp.h>
using namespace std;
void merge(int* numbers, int low, int high){
	int mid = (low+high)/2;
	int i = low;
	int j = mid+1;

	int* aux = new int[high-low+1];
	int k = 0;

	while(i <= mid && j <= high)
		aux[k++] = numbers[i] < numbers[j] ? numbers[i++] : numbers[j++];
	while(i <= mid)
		aux[k++] = numbers[i++];
	while(j <= high)
		aux[k++] = numbers[j++];

	for(int k = 0; k < (high-low+1); k++)
		numbers[low+k] = aux[k];
}
void up_mergeSort(int* numbers, int low, int high){
	if(low >= high)
		return;
	int mid = (low+high)/2;
	up_mergeSort(numbers, low, mid);
	up_mergeSort(numbers, mid+1, high);
	merge(numbers, low, high);
}
void p_mergeSort(int* numbers, int low, int high){
	if(low >= high)
		return;
	int mid = (low+high)/2;
	
	#pragma omp parallel sections
	{	
		#pragma omp section
		{
			p_mergeSort(numbers, low, mid);
		}
		#pragma omp section
		{
			p_mergeSort(numbers, mid+1, high);
		}
	}

	merge(numbers, low, high);
}
int main()
{

	int size;
	cout << "Enter size : ";
	cin >> size;

	int up_numbers[size];
	int p_numbers[size];
	for(int i = 0; i < size; i++){
		up_numbers[i] = rand()%(2*size)+1;
		p_numbers[i] = up_numbers[i];
	}

	clock_t totalTime = clock();
	up_mergeSort(up_numbers, 0, size-1);
	totalTime = clock()-totalTime;

	cout << "Total time required (Unparallel merge sort) : " <<  float(totalTime)/CLOCKS_PER_SEC*1000 << " milliseconds" << endl;

	totalTime = clock();
	p_mergeSort(p_numbers, 0, size-1);
	totalTime = clock()-totalTime;

	cout << "Total time required (Parallel merge sort) : " <<  float(totalTime)/CLOCKS_PER_SEC*1000 << " milliseconds" << endl;
}
