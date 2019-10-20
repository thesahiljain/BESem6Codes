#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void min(int* input)
{
	int tid = threadIdx.x;
	float number_of_threads = blockDim.x;
	int step_size = 1;

	while(number_of_threads > 0){
		if(tid < number_of_threads)
		{
			int first = tid*step_size*2;
			int second = first + step_size;
			if(input[first] > input[second] && input[second] > 0)
				input[first] = input[second];
		}
		step_size *= 2;
		number_of_threads = number_of_threads!=1 ? (int)ceil(number_of_threads/2) : 0;
	}
}

int main(int argc, char const *argv[])
{
    // User input
	int count;
	cout << "Enter size : ";
	cin >> count;

    // Host array
	int hostArray[count];
	for (int i = 0; i < count; i++)
		hostArray[i] = rand()%count+1;

    // Device array
	int *deviceArray;
	cudaMalloc(&deviceArray, count*sizeof(int));
	cudaMemcpy(deviceArray, hostArray, count*sizeof(int), cudaMemcpyHostToDevice);

    // Cuda code
	min<<<1, (count/2)+1>>>(deviceArray);
	int result;
	cudaMemcpy(&result, deviceArray, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Elements : ";
	for(int i = 0; i < count; i++)
        cout << hostArray[i] << " ";
	cout << "\nMinimum element: " << result;
}
/*
Enter size : 25
Elements : 9 12 3 16 19 11 12 18 25 22 13 3 16 10 14 2 16 2 23 12 12 19 18 5 8
Minimum element: 2
*/
