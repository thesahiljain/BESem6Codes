#include <iostream>
#include <stdio.h>
#include <math.h>
using namespace std;

__global__ void sum(float* input)
{
	int tid = threadIdx.x;
	float number_of_threads = blockDim.x;
	int step_size = 1;

	while(number_of_threads > 0){
		if(tid < number_of_threads)
		{
			int first = tid*step_size*2;
			int second = first + step_size;
            input[first] += input[second];
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
	float hostArray[count];
	for (int i = 0; i < count; i++)
		hostArray[i] = rand()%count+1;

    // Device array
	float *deviceArray;
	cudaMalloc(&deviceArray, count*sizeof(float));
	cudaMemcpy(deviceArray, hostArray, count*sizeof(float), cudaMemcpyHostToDevice);

    // Cuda code
	sum<<<1, (count/2)+1>>>(deviceArray);
	float mean;
	cudaMemcpy(&mean, deviceArray, sizeof(float), cudaMemcpyDeviceToHost);
    mean = (float)mean/count;

	cout << "Elements : ";
	for(int i = 0; i < count; i++)
        cout << hostArray[i] << " ";
	cout << "\nArithmetic mean: " << mean << endl;

	// Recalculation
    for(int i = 0; i < count; i++)
        hostArray[i] = (hostArray[i]-mean)*(hostArray[i]-mean);
    cudaMemcpy(deviceArray, hostArray, count*sizeof(float), cudaMemcpyHostToDevice);

    sum<<<1, (count/2)+1>>>(deviceArray);
    float variance;
    cudaMemcpy(&variance, deviceArray, sizeof(float), cudaMemcpyDeviceToHost);
    variance = (float)variance/count;

    cout << "Standard deviation : " << sqrt(variance) << endl;
}
/*
Enter size : 3
Elements : 2 2 1
Arithmetic mean: 1.66667
Standard deviation : 0.471404
*/