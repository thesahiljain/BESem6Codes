#include <iostream>
#include <stdio.h>
using namespace std;

__global__ void max(int* input)
{
	int tid = threadIdx.x;
	float number_of_threads = blockDim.x;
	int step_size = 1;

	while(number_of_threads > 0){
		if(tid < number_of_threads)
		{
			int first = tid*step_size*2;
			int second = first + step_size;
			if(input[first] < input[second] && input[second] > 0)
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
	max<<<1, (count/2)+1>>>(deviceArray);
	int result;
	cudaMemcpy(&result, deviceArray, sizeof(int), cudaMemcpyDeviceToHost);

	cout << "Elements : ";
	for(int i = 0; i < count; i++)
        cout << hostArray[i] << " ";
	cout << "\nMaximum element: " << result;
}
/*
Enter size : 30
Elements : 14 17 28 26 24 26 17 13 10 2 3 8 21 20 24 17 1 7 23 17 12 9 28 10 3 21 3 14 8 26
Maximum element: 28
*/
