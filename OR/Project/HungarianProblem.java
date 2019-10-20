import java.util.*;

public class HungarianProblem {
	
	static Scanner input = new Scanner(System.in);
	static int cost[][];
	static int size;
	static HashMap<Integer, Integer> assignments = new HashMap<>();
	static HashMap<Integer, Integer> cancelled = new HashMap<>();
	
	public static void main(String[] args) {
		
		System.out.print("Enter number of operators/machines : ");
		size = input.nextInt();
		cost = new int[size][size];
		
		System.out.println("Enter the elements of cost matrix");
		for(int i = 0; i < size; i++)
			for(int j = 0; j < size; j++)
				cost[i][j] = input.nextInt();
		
		rowOperation();
		columnOperation();
		
		while(assignments.size() != size) {
			rowScanning();
			columnScanning();
		}
	}
	
	static void rowOperation() {
		for(int i = 0; i < size; i++) {
			int min = Integer.MAX_VALUE;
			for(int j = 0; j < size; j++)
				min = Math.min(min, cost[i][j]);
			for(int j = 0; j < size; j++)
				cost[i][j] = cost[i][j] - min;
		}
		System.out.println("Row operation result");
		display();
	}
	
	static void columnOperation() {
		for(int j = 0; j < size; j++) {
			int min = Integer.MAX_VALUE;
			for(int i = 0; i < size; i++)
				min = Math.min(min, cost[i][j]);
			for(int i = 0; i < size; i++)
				cost[i][j] = cost[i][j] - min;
		}
		System.out.println("Column operation result");
		display();
	}
	
	static void rowScanning() {
		for(int row = 0; row < size; row++) {
			if(zerosRow(row) != 1) continue;
			int col;
			for(col = 0; col < size; col++)
				if(cost[row][col] == 0)
					break;
			assignments.put(row, col);
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
			if(zerosColumn(col) != 1) continue;
			int row;
			for(row = 0; row < size; row++)
				if(cost[row][col] == 0)
					break;
			assignments.put(row, col);
			for(int j = 0; j < size; j++) {
				if(col == j) continue;
				if(cost[row][j] == 0) cancelled.put(row, j);
			}
		}
		
		System.out.println("Column scanning");
		display();
	}
	
	static void display() {
		for(int i = 0; i < size; i++) {
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
	
	static int zerosRow(int row) {
		int zeros = 0;
		for(int j = 0; j < size; j++) 
			if(cost[row][j] == 0 && assignments.get(row) != Integer.valueOf(j) && cancelled.get(row) != Integer.valueOf(j))
				zeros++;
		return zeros;
	}
	
	static int zerosColumn(int col) {
		int zeros = 0;
		for(int i = 0; i < size; i++)
			if(cost[i][col] == 0 && assignments.get(i) != Integer.valueOf(col) && cancelled.get(i) != Integer.valueOf(col))
				zeros++;
		return zeros;
	}
}

/*
4
8 26 17 11
13 28 4 26
38 19 18 15
19 26 24 10
*/