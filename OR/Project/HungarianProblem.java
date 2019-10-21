import java.util.*;

public class Main {
	
	static final String[] workers = new String[]{"I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"};
	static final String[] machines = new String[] {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};
	static Scanner input = new Scanner(System.in);
	static int cost[][];
	static int size;
	static HashMap<Integer, Integer> assignments = new HashMap<>();
	static HashMap<Integer, Integer> cancelled = new HashMap<>();
	
	public static void main(String[] args) {
		
		// Accept user input and store an original copy for later
		System.out.print("Enter number of operators/machines : ");
		size = input.nextInt();
		cost = new int[size][size];
 		int original[][] = new int[size][size];
		
		System.out.println("Enter the elements of cost matrix");
		for(int i = 0; i < size; i++)
			for(int j = 0; j < size; j++) {
				cost[i][j] = input.nextInt();
				original[i][j] = cost[i][j];
			}
		System.out.println();
		
		// Perform row and column operation
		rowOperation();
		columnOperation();
		
		// Repeat row scanning and column scanning till all assignments are done
		while(assignments.size() != size) {
			rowScanning();
			columnScanning();
		}
		
		// Display final allocations and total cost
		System.out.println("Final Allocations");
		for(int worker : assignments.keySet())
			System.out.println(workers[worker] + "\t:\t" + machines[assignments.get(worker)]);
		
		int totalCost = 0;
		for(int i : assignments.keySet())
			totalCost += original[i][assignments.get(i)];
		System.out.println("Total Cost : " + totalCost);
		
	}
	
	static void rowOperation() {
		for(int i = 0; i < size; i++) {
			// Find minimum value of row
			int min = Integer.MAX_VALUE;
			for(int j = 0; j < size; j++)
				min = Math.min(min, cost[i][j]);
			// Subtract minimum value from each element of row
			for(int j = 0; j < size; j++)
				cost[i][j] = cost[i][j] - min;
		}
		System.out.println("Row operation result");
		display();
	}
	
	static void columnOperation() {
		for(int j = 0; j < size; j++) {
			int min = Integer.MAX_VALUE;
			// Find minimum value of column
			for(int i = 0; i < size; i++)
				min = Math.min(min, cost[i][j]);
			// Subtract minimum value from each element of column
			for(int i = 0; i < size; i++)
				cost[i][j] = cost[i][j] - min;
		}
		System.out.println("Column operation result");
		display();
	}
	
	static void rowScanning() {
		for(int row = 0; row < size; row++) {
			// If the number of remaining zeros is not 1, skip the row
			if(zerosRow(row) != 1) continue;
			int col;
			for(col = 0; col < size; col++)
				if(cost[row][col] == 0)
					break;
			// Assign the row and column
			assignments.put(row, col);
			// Cancel other zeros in the column
			for(int i = 0; i < size; i++) {
				if(row == i) continue;
				if(cost[i][col] == 0) cancelled.put(i, col);
			}
		}
		System.out.println("Row scanning");
		display();
	}
	
	static void columnScanning() {
		for(int col = 0; col < size; col++) {
			// If number of zeros in column is not 1, skip column
			if(zerosColumn(col) != 1) continue;
			int row;
			for(row = 0; row < size; row++)
				if(cost[row][col] == 0)
					break;
			// Assign the row and column
			assignments.put(row, col);
			// Cancel remaining zeros in the row
			for(int j = 0; j < size; j++) {
				if(col == j) continue;
				if(cost[row][j] == 0) cancelled.put(row, j);
			}
		}
		
		System.out.println("Column scanning");
		display();
	}
	
	// Method to display the matrix with appropriate cancellations and assignments
	static void display() {
		System.out.print("\t");
		for(int i = 0; i < size; i++)
			System.out.print(machines[i] + "\t");
		System.out.println();
		for(int i = 0; i < size; i++) {
			System.out.print(workers[i] + "\t");
			for(int j = 0; j < size; j++) {
				if(assignments.get(i) == Integer.valueOf(j))
					System.out.print("|0|");
				else if(cancelled.get(i) == Integer.valueOf(j))
					System.out.print("X");
				else
					System.out.print(cost[i][j]);
				System.out.print("\t");
			}
			System.out.println();
		}
		System.out.println();
	}
	
	// Method to count zeros in the rows
	static int zerosRow(int row) {
		int zeros = 0;
		for(int j = 0; j < size; j++) 
			// Skip assigned and cancelled zeros
			if(cost[row][j] == 0 && assignments.get(row) != Integer.valueOf(j) && cancelled.get(row) != Integer.valueOf(j))
				zeros++;
		return zeros;
	}
	// Method to count zeros in the column
	static int zerosColumn(int col) {
		int zeros = 0;
		for(int i = 0; i < size; i++)
			// Skip assigned and cancelled zeros
			if(cost[i][col] == 0 && assignments.get(i) != Integer.valueOf(col) && cancelled.get(i) != Integer.valueOf(col))
				zeros++;
		return zeros;
	}
}

/*
Enter number of operators/machines : 4
Enter the elements of cost matrix
8 26 17 11
13 28 4 26
38 19 18 15
19 26 24 10

Row operation result
	A	B	C	D	
I	0	18	9	3	
II	9	24	0	22	
III	23	4	3	0	
IV	9	16	14	0	

Column operation result
	A	B	C	D	
I	0	14	9	3	
II	9	20	0	22	
III	23	0	3	0	
IV	9	12	14	0	

Row scanning
	A	B	C	D	
I	|0|	14	9	3	
II	9	20	|0|	22	
III	23	0	3	X	
IV	9	12	14	|0|	

Column scanning
	A	B	C	D	
I	|0|	14	9	3	
II	9	20	|0|	22	
III	23	|0|	3	X	
IV	9	12	14	|0|	

Final Allocations
I	:	A
II	:	C
III	:	B
IV	:	D
Total Cost : 41
*/