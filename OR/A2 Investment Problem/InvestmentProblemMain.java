
public class InvestmentProblemMain {
	
	static double opportunities[][] = new double[][]
			{{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        	{4.1, 1.8, 1.5, 2.2, 1.3, 4.2, 2.2, 1.0, 0.5},
        	{5.8, 3.0, 2.5, 3.8, 2.4, 5.9, 3.5, 1.7, 1.0},
        	{6.5, 3.9, 3.3, 4.8, 3.2, 6.6, 4.2, 2.3, 1.5},
        	{6.8, 4.5, 3.8, 5.5, 3.9, 6.8, 4.6, 2.8, 2}};
    static double revenues[][] = new double[11][9];
	
	public static void main(String[] args) {
		
		// Revenue matrix
		System.out.println("Opportunity matrix");
		display(opportunities);
		
		// Revenue calculation
		for(int i = 0; i < 11; i++)
		for(int j = 8; j >= 0; j--) {
			// Case where no money is invested
			if(i == 0)
				revenues[i][j] = 0;
			// Case where only no opportunity is used
			else if(j == 8)
				revenues[i][j] = revenues[i-1][j]+0.5;
			// All other cases
			else {
				revenues[i][j] = Double.MIN_VALUE;
				for(int unit = 0; unit <= 4 && unit <= i; unit++)
					revenues[i][j] = Math.max(revenues[i][j], opportunities[unit][j]+revenues[i-unit][j+1]);
			}
		}
		
		// Show results
		System.out.println("Revenue matrix");
		display(revenues);
		
		System.out.printf("Maximum profit : %.2f\n", revenues[10][0]);
		
	}
	
	static void display(double[][] matrix) {
		for(double[] row : matrix) {
			for(double element : row)
				System.out.printf("%.2f\t", element);
			System.out.println();
		}
		System.out.println();
	}
	
}
