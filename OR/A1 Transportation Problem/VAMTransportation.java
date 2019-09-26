import java.util.*;

public class Main {
	
	static Scanner input = new Scanner(System.in);
	
	static int[][] matrix = new int[][] {{16, 18, 21, 12}, {17, 19, 14, 13}, {32, 11, 15, 10}};
	static int[] supply = new int[] {150, 160, 90};
	static int[] capacity = new int[] {140, 120, 90, 50};
	static int[][] allocation = new int[3][4];
	
	public static void main(String[] args) {
		display(matrix);
		if(isBalanced())
			System.out.println("The given transportation problem is balanced\n");
		else {
			System.out.println("The given transportation problem is unbalanced\n");
			return;
		}
		
		int iteration = 0;
		
		while(!isAllocationComplete()) {
			iteration++;
			
			int row = getOptimumLocation()[0];
			int col = getOptimumLocation()[1];
			
			int allocationValue = Math.min(supply[row], capacity[col]);
			supply[row] -= allocationValue;
			capacity[col] -= allocationValue;
			allocation[row][col] = allocationValue;
			
			System.out.println("Iteration #" + iteration + "\n");
			System.out.println("Cost Matrix");
			display(matrix);
			System.out.println("Allocation Matrix");
			display(allocation);
		}
		
		showTotalCost();
	}
	
	static int[] getOptimumLocation(){
		int row = 0, col = 0;
		int rowPenalty[] = rowPenalty();
		int colPenalty[] = colPenalty();

		int rowMaxIndex = 0, colMaxIndex = 0;
		for(int i = 0; i < rowPenalty.length; i++)
			if(rowPenalty[i] > rowPenalty[rowMaxIndex])
				rowMaxIndex = i;
		for(int i = 0; i < colPenalty.length; i++)
			if(colPenalty[i] > colPenalty[colMaxIndex])
				colMaxIndex = i;
		
		if(rowPenalty[rowMaxIndex] > colPenalty[colMaxIndex]){
			row = rowMaxIndex;
			int minimum = Integer.MAX_VALUE;
			for(int j = 0; j < capacity.length; j++){
				if(capacity[j] != 0){
					if(matrix[row][j] < minimum){
						minimum = matrix[row][j];
						col = j;
					}
				}
			}
		}else{
			col = colMaxIndex;
			int minimum = Integer.MAX_VALUE;
			for(int j = 0; j < supply.length; j++){
				if(supply[j] != 0){
					if(matrix[j][col] < minimum){
						minimum = matrix[j][col];
						row = j;
					}
				}
			}
		}
		
		return new int[]{row, col};
	}
	
	static int[] rowPenalty(){
		int[] rowPenalty = new int[supply.length];
		for(int i = 0; i < rowPenalty.length; i++){
			if(supply[i] == 0){
				rowPenalty[i] = Integer.MIN_VALUE;
				continue;
			}
			int freeCells = 0;
			for(int j = 0; j < capacity.length; j++) if(capacity[j] != 0) freeCells++;
			if(freeCells <= 1){
				rowPenalty[i] = 0;
				continue;
			}
			int minimum = Integer.MAX_VALUE;
			for(int j = 0; j < capacity.length; j++)
				if(capacity[j] != 0)
					minimum = Math.min(minimum, matrix[i][j]);
			int secondMinimum = Integer.MAX_VALUE;
			for(int j = 0; j < capacity.length; j++)
				if(capacity[j] != 0 && matrix[i][j] > minimum)
					secondMinimum = Math.min(secondMinimum, matrix[i][j]);
			rowPenalty[i] = secondMinimum-minimum;
		}
		return rowPenalty;
	}
	
	static int[] colPenalty(){
		int[] colPenalty = new int[capacity.length];
		for(int j = 0; j < colPenalty.length; j++){
			if(capacity[j] == 0){
				colPenalty[j] = Integer.MIN_VALUE;
				continue;
			}
			int freeCells = 0;
			for(int i = 0; i < supply.length; i++) if(supply[i] != 0) freeCells++;
			if(freeCells <= 1){
				colPenalty[j] = 0;
				continue;
			}
			int minimum = Integer.MAX_VALUE;
			for(int i = 0; i < supply.length; i++)
				if(supply[i] != 0)
					minimum = Math.min(minimum, matrix[i][j]);
			int secondMinimum = Integer.MAX_VALUE;
			for(int i = 0; i < supply.length; i++)
				if(matrix[i][j] > minimum && supply[i] != 0)
					secondMinimum = Math.min(secondMinimum, matrix[i][j]);
			colPenalty[j] = secondMinimum-minimum;
		}
		return colPenalty;
	}
	
	static void display(int[][] matrix) {
		
		int[] rowPenalty = rowPenalty();
		int[] colPenalty = colPenalty();
		
		System.out.println("\tP\tQ\tR\tS\tSupply\tPenalty");
		String[] routes = new String[] {"A", "B", "C"};
		
		for(int i = 0; i < 3; i++) {
			System.out.print(routes[i] + "\t");
			for(int j = 0; j < 4; j++)
				System.out.print(matrix[i][j] + "\t");
			System.out.print(supply[i] + "\t");
			if(rowPenalty[i] != Integer.MIN_VALUE)
				System.out.println(rowPenalty[i]);
			else
				System.out.println("--");
		}
		
		System.out.print("Demand\t");
		for(int i = 0; i < 4; i++)
			System.out.print(capacity[i] + "\t");
		System.out.println();
		
		System.out.print("Penalty\t");
		for(int i = 0; i < 4; i++){
			if(colPenalty[i] != Integer.MIN_VALUE)
				System.out.print(colPenalty[i] + "\t");
			else
				System.out.print("--\t");
		}
		System.out.println("Optimal Location : " + getOptimumLocation()[0] + ", " + getOptimumLocation()[1]);
		System.out.println();
	}
	
	static boolean isBalanced() {
		int sumOfSupply = 0, sumOfCapacity = 0;
		for(int x : supply) sumOfSupply += x;
		for(int x : capacity) sumOfCapacity += x;
		return sumOfSupply == sumOfCapacity;
	}
	
	static boolean isAllocationComplete() {
		for(int x : supply)
			if(x != 0) return false;
		for(int x : capacity)
			if(x != 0) return false;
		return true;
	}
	
	static void showTotalCost() {
		int total = 0;
		for(int i = 0; i < 3; i++)
		for(int j = 0; j < 4; j++)
			total += matrix[i][j]*allocation[i][j];
		System.out.println("Total Cost of Transportation is " + total);
	}
}
/*
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	150	4
B	17	19	14	13	160	1
C	32	11	15	10	90	1
Demand	140	120	90	50	
Penalty	1	7	1	2	Optimal Location : 2, 1

The given transportation problem is balanced

Iteration #1

Cost Matrix
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	150	4
B	17	19	14	13	160	1
C	32	11	15	10	0	--
Demand	140	30	90	50	
Penalty	1	1	7	1	Optimal Location : 1, 2

Allocation Matrix
	P	Q	R	S	Supply	Penalty
A	0	0	0	0	150	4
B	0	0	0	0	160	1
C	0	90	0	0	0	--
Demand	140	30	90	50	
Penalty	1	1	7	1	Optimal Location : 1, 2

Iteration #2

Cost Matrix
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	150	4
B	17	19	14	13	70	4
C	32	11	15	10	0	--
Demand	140	30	0	50	
Penalty	1	1	--	1	Optimal Location : 0, 3

Allocation Matrix
	P	Q	R	S	Supply	Penalty
A	0	0	0	0	150	4
B	0	0	90	0	70	4
C	0	90	0	0	0	--
Demand	140	30	0	50	
Penalty	1	1	--	1	Optimal Location : 0, 3

Iteration #3

Cost Matrix
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	100	2
B	17	19	14	13	70	2
C	32	11	15	10	0	--
Demand	140	30	0	0	
Penalty	1	1	--	--	Optimal Location : 0, 0

Allocation Matrix
	P	Q	R	S	Supply	Penalty
A	0	0	0	50	100	2
B	0	0	90	0	70	2
C	0	90	0	0	0	--
Demand	140	30	0	0	
Penalty	1	1	--	--	Optimal Location : 0, 0

Iteration #4

Cost Matrix
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	0	--
B	17	19	14	13	70	2
C	32	11	15	10	0	--
Demand	40	30	0	0	
Penalty	0	0	--	--	Optimal Location : 1, 0

Allocation Matrix
	P	Q	R	S	Supply	Penalty
A	100	0	0	50	0	--
B	0	0	90	0	70	2
C	0	90	0	0	0	--
Demand	40	30	0	0	
Penalty	0	0	--	--	Optimal Location : 1, 0

Iteration #5

Cost Matrix
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	0	--
B	17	19	14	13	30	0
C	32	11	15	10	0	--
Demand	0	30	0	0	
Penalty	--	0	--	--	Optimal Location : 1, 1

Allocation Matrix
	P	Q	R	S	Supply	Penalty
A	100	0	0	50	0	--
B	40	0	90	0	30	0
C	0	90	0	0	0	--
Demand	0	30	0	0	
Penalty	--	0	--	--	Optimal Location : 1, 1

Iteration #6

Cost Matrix
	P	Q	R	S	Supply	Penalty
A	16	18	21	12	0	--
B	17	19	14	13	0	--
C	32	11	15	10	0	--
Demand	0	0	0	0	
Penalty	--	--	--	--	Optimal Location : 0, 0

Allocation Matrix
	P	Q	R	S	Supply	Penalty
A	100	0	0	50	0	--
B	40	30	90	0	0	--
C	0	90	0	0	0	--
Demand	0	0	0	0	
Penalty	--	--	--	--	Optimal Location : 0, 0

Total Cost of Transportation is 5700

*/