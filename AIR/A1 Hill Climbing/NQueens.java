import java.util.*;

public class Main {
	
	static int[] board;
	static int steps = 0;
	static int dimension;
	
	public static void main(String[] args) {
		
		System.out.print("Enter dimension : ");
		dimension = new Scanner(System.in).nextInt();
		if(dimension == 2 || dimension == 3 || dimension <= 0) {
			System.out.println("Doesn't work for dimension "+ dimension);
			return;
		}
		
		board = generateBoard();
		
		displayBoard(board);
		while(findHeuristic(board) != 0) {
			nextBoard();
			displayBoard(board);
		}
		System.out.println();
		
		int[][] map = new int[dimension][dimension];
		for(int i = 0; i < dimension; i++)
			map[i][board[i]] = 1;
		for(int i = 0; i < dimension; i++) {
			for(int j = 0; j < dimension; j++)
				System.out.print(map[i][j] + " ");
			System.out.println();
		}
	}
	
	public static int[] generateBoard() {
		int[] board = new int[dimension];
		Random random = new Random();
		for(int i = 0; i < dimension; i++)
			board[i] = random.nextInt(dimension);
		return board;
	}
	
	public static void displayBoard(int[] board) {
		System.out.print("Board : ");
		for(int i = 0; i < dimension; i++)
			System.out.print("(" + (i+1) + ", " + (board[i]+1) + ") ");
		System.out.println("Heuristic : " + findHeuristic(board) + " Steps : " + steps);
	}
	
	public static int findHeuristic(int[] board) {
		int heuristic = 0;
		for(int i = 0; i < dimension; i++)
			for(int j = i+1; j < dimension; j++)
				if(board[i] == board[j] || Math.abs(i-j) == Math.abs(board[i]-board[j]))
					heuristic++;
		return heuristic;
	}
	
	public static void nextBoard() {
		steps++;
		int[] temp = board.clone();
		for(int i = 0; i < dimension; i++) {
			temp = board.clone();
			temp[i] = (temp[i]+1)%dimension;
			if(findHeuristic(temp) < findHeuristic(board))
				break;
		}
		if(findHeuristic(temp) >= findHeuristic(board))
			board = generateBoard();
		else
			board = temp;
	}
}