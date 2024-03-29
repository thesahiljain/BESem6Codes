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
		int row = 0, col = 0;
		
		while(!isAllocationComplete()) {
			iteration++;
			
			int allocationValue = Math.min(supply[row], capacity[col]);
			supply[row] -= allocationValue;
			capacity[col] -= allocationValue;
			allocation[row][col] = allocationValue;
			
			System.out.println("Iteration #" + iteration + "\n");
			System.out.println("Cost Matrix");
			display(matrix);
			System.out.println("Allocation Matrix");
			display(allocation);
			
			if(supply[row] == 0) row++;
			else col++;
		}
		
		showTotalCost();
	}
	
	static void display(int[][] matrix) {
		System.out.println("\tP\tQ\tR\tS\tSupply");
		String[] routes = new String[] {"A", "B", "C"};
		
		for(int i = 0; i < 3; i++) {
			System.out.print(routes[i] + "\t");
			for(int j = 0; j < 4; j++)
				System.out.print(matrix[i][j] + "\t");
			System.out.println(supply[i]);
		}
		
		System.out.print("Demand\t");
		for(int i = 0; i < 4; i++)
			System.out.print(capacity[i] + "\t");
		System.out.println("\n");
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
	P	Q	R	S	Supply
A	16	18	21	12	150
B	17	19	14	13	160
C	32	11	15	10	90
Demand	140	120	90	50	

The given transportation problem is balanced

Iteration #1

Cost Matrix
	P	Q	R	S	Supply
A	16	18	21	12	10
B	17	19	14	13	160
C	32	11	15	10	90
Demand	0	120	90	50	

Allocation Matrix
	P	Q	R	S	Supply
A	140	0	0	0	10
B	0	0	0	0	160
C	0	0	0	0	90
Demand	0	120	90	50	

Iteration #2

Cost Matrix
	P	Q	R	S	Supply
A	16	18	21	12	0
B	17	19	14	13	160
C	32	11	15	10	90
Demand	0	110	90	50	

Allocation Matrix
	P	Q	R	S	Supply
A	140	10	0	0	0
B	0	0	0	0	160
C	0	0	0	0	90
Demand	0	110	90	50	

Iteration #3

Cost Matrix
	P	Q	R	S	Supply
A	16	18	21	12	0
B	17	19	14	13	50
C	32	11	15	10	90
Demand	0	0	90	50	

Allocation Matrix
	P	Q	R	S	Supply
A	140	10	0	0	0
B	0	110	0	0	50
C	0	0	0	0	90
Demand	0	0	90	50	

Iteration #4

Cost Matrix
	P	Q	R	S	Supply
A	16	18	21	12	0
B	17	19	14	13	0
C	32	11	15	10	90
Demand	0	0	40	50	

Allocation Matrix
	P	Q	R	S	Supply
A	140	10	0	0	0
B	0	110	50	0	0
C	0	0	0	0	90
Demand	0	0	40	50	

Iteration #5

Cost Matrix
	P	Q	R	S	Supply
A	16	18	21	12	0
B	17	19	14	13	0
C	32	11	15	10	50
Demand	0	0	0	50	

Allocation Matrix
	P	Q	R	S	Supply
A	140	10	0	0	0
B	0	110	50	0	0
C	0	0	40	0	50
Demand	0	0	0	50	

Iteration #6

Cost Matrix
	P	Q	R	S	Supply
A	16	18	21	12	0
B	17	19	14	13	0
C	32	11	15	10	0
Demand	0	0	0	0	

Allocation Matrix
	P	Q	R	S	Supply
A	140	10	0	0	0
B	0	110	50	0	0
C	0	0	40	50	0
Demand	0	0	0	0	

Total Cost of Transportation is 6310
*/
