#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int main(int argc, char* argv[]){

	int a[] = {1, 3, 12, 21, 91, 2};
	
	// Initialize the MPI environment
	MPI_Init(NULL, NULL);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// Check for minimum 2 processes
	if (size < 2){
		cout << "Error : World size must be greater than 1 for " << argv[0] << endl; 
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	float localSum;
	float globalSum;
	float globalMinima;
	float globalMaxima;
	localSum = (float)a[rank];
	
	// Reduction
	MPI_Reduce(&localSum, &globalSum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&localSum, &globalMinima, 1, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&localSum, &globalMaxima, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
	
	// Display results
	if (rank == 0)
	{
		cout << "Total sum : " << globalSum << " Average : " << globalSum/(size+1) << endl;
		cout << "Minima : " << globalMinima << endl;
		cout << "Maxima : " << globalMaxima << endl;
	}
	MPI_Finalize();
}
