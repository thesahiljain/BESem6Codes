import java.util.*;

public class Main {
	
	static Scanner input = new Scanner(System.in);
	
	public static void main(String[] args) {
		Node start = new Node(0);
		System.out.println("Enter the initial state");
		for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			start.matrix[i][j] = input.nextInt();
		
		PriorityQueue<Node> OPEN = new PriorityQueue<>(new Comparator<Node>() {
			public int compare(Node o1, Node o2) {
				if(o1.getCost() != o2.getCost())
					return o1.getCost()-o2.getCost();
				return o1.getHeuristic()-o2.getCost();
			}
		});
		
		List<Node> CLOSE = new ArrayList<>();
		Stack<Node> solution = new Stack<>();
		
		OPEN.add(start);
		
		boolean found = false;
		while(!OPEN.isEmpty()) {
			Node current = OPEN.poll();
			if(current.getHeuristic() == 0) {
				while(current != null) {
					solution.add(current);
					current = current.parent;
				}
				break;
			}
			CLOSE.add(current);
			
			for(Node child : current.getMoves()) {
				boolean isClosed = false;
				for(Node node : CLOSE)
					if(child.isSame(node)) {
						isClosed = true;
						break;
					}
				for(Node node : OPEN) {
					if(child.isSame(node) && (child.getCost() < node.getCost())) {
						Node parent = child.parent;
						child.parent = node.parent;
						node.parent = parent;
						
						int height = child.height;
						child.height = node.height;
						node.height = height;
						isClosed = true;
						break;
					}
				}
				if(!isClosed)
					OPEN.add(child);
			}
		}
		if(solution.isEmpty())
			System.out.println("Solution not found!");
		else {
			System.out.println("Solution found!");
			while(!solution.isEmpty())
				solution.pop().display();
		}
	}
}

class Node{
	
	public int[][] matrix = new int[3][3];
	public Node parent = null;
	public int height;
	
	public Node(int height) {this.height = height;}
	
	public int getHeuristic() {
		int heuristic = 0;
		
		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++) {
				switch(matrix[i][j]) {
				case 1:
					heuristic += Math.abs(i-0)+Math.abs(j-0);
					break;
				case 2:
					heuristic += Math.abs(i-0)+Math.abs(j-1);
					break;
				case 3:
					heuristic += Math.abs(i-0)+Math.abs(j-2);
					break;
				case 4:
					heuristic += Math.abs(i-1)+Math.abs(j-0);
					break;
				case 5:
					heuristic += Math.abs(i-1)+Math.abs(j-1);
					break;
				case 6:
					heuristic += Math.abs(i-1)+Math.abs(j-2);
					break;
				case 7:
					heuristic += Math.abs(i-2)+Math.abs(j-0);
					break;
				case 8:
					heuristic += Math.abs(i-2)+Math.abs(j-1);
					break;
				default:
					break;
				}
			}
		
		return heuristic;
	}
	
	public int getCost() {return height+getHeuristic();}
	
	public void display() {
		for(int i = 0; i < 3; i++) {
			for(int j = 0; j < 3; j++) {
				if(matrix[i][j] == 0) {
					System.out.print("_" + "\t");
					continue;
				}
				System.out.print(matrix[i][j] + "\t");
			}
			System.out.println();
		}
		System.out.println("Height : " + height + " Heuristic : " + getHeuristic() + " Cost : " + getCost() + "\n");
	}
	
	List<Node> getMoves(){
		ArrayList<Node> list = new ArrayList<>();
		int row=0, col=0;

		for(int i = 0; i < 3; i++)
			for(int j = 0; j < 3; j++)
				if(matrix[i][j] == 0) {
					row = i;
					col = j;
					break;
				}
		
		if(row > 0) {
			Node move1 = getCopy();
			move1.height++;
			move1.parent = this;
			int t = move1.matrix[row][col];
			move1.matrix[row][col] = move1.matrix[row-1][col];
			move1.matrix[row-1][col] = t;
			list.add(move1);
		}
		if(row < 2) {
			Node move2 = getCopy();
			move2.height++;
			move2.parent = this;
			int t = move2.matrix[row][col];
			move2.matrix[row][col] = move2.matrix[row+1][col];
			move2.matrix[row+1][col] = t;
			list.add(move2);
		}
		if(col > 0) {
			Node move3 = getCopy();
			move3.height++;
			move3.parent = this;
			int t = move3.matrix[row][col];
			move3.matrix[row][col] = move3.matrix[row][col-1];
			move3.matrix[row][col-1] = t;
			list.add(move3);
		}
		if(col < 2) {
			Node move4 = getCopy();
			move4.height++;
			move4.parent = this;
			int t = move4.matrix[row][col];
			move4.matrix[row][col] = move4.matrix[row][col+1];
			move4.matrix[row][col+1] = t;
			list.add(move4);
		}
		return list;
	}
	
	Node getCopy() {
		Node node = new Node(height);
		for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			node.matrix[i][j] = matrix[i][j];
		return node;
	}
	
	boolean isSame(Node node) {
		for(int i = 0; i < 3; i++)
		for(int j = 0; j < 3; j++)
			if(matrix[i][j] != node.matrix[i][j])
				return false;
		return true;
	}
}
