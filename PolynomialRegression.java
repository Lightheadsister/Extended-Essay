import java.util.ArrayList;
import java.util.List;

public class PolynomialRegression {

    private static final int N = 1000; 
    private static final int DEGREE = 10; 
    protected double[] coefficients;


    private List<Double> xData = new ArrayList<>();
    private List<Double> yData = new ArrayList<>();
    
    
    public PolynomialRegression() {
        coefficients = new double[DEGREE + 1];
    }

    public double function(double x) {
        double y = 2 * Math.pow(x, 3) + 3 * Math.pow(x, 2) + 4 * x;
     
        if (x % 2 == 0) {
            y = y + 1 ;
        } else {
            y = y - 1; 
        }
        return y;
    }
    
    
    public void addTrainingData(double x, double y) {
        xData.add(x);
        yData.add(y);
    }
    
    
    public void train() {
        double[][] matrix = new double[DEGREE + 1][DEGREE + 2];
        for (int i = 0; i <= DEGREE; i++) {
            for (int j = 0; j <= DEGREE; j++) {
                matrix[i][j] = 0;
                for (int k = 0; k < N; k++) {
                    double x = k + 1;
                    matrix[i][j] += Math.pow(x, i + j);
                }
            }
        }

        double[] Y = new double[DEGREE + 1];
        for (int i = 0; i <= DEGREE; i++) {
            Y[i] = 0;
            for (int j = 0; j < N; j++) {
                double x = j + 1;
                Y[i] += Math.pow(x, i) * function(x);
            }
        }

        for (int i = 0; i <= DEGREE; i++) {
            matrix[i][DEGREE + 1] = Y[i];
        }

        coefficients = gaussianElimination1(matrix);
    }

    private double[] gaussianElimination1(double[][] matrix) {
        int n = matrix.length;
        for (int p = 0; p < n; p++) {
            int max = p;
            for (int i = p + 1; i < n; i++) {
                if (Math.abs(matrix[i][p]) > Math.abs(matrix[max][p])) {
                    max = i;
                }
            }

            double[] temp = matrix[p];
            matrix[p] = matrix[max];
            matrix[max] = temp;

            if (Math.abs(matrix[p][p]) <= 1e-10) {
                throw new RuntimeException("Matrix is singular or nearly singular");
            }

            for (int i = p + 1; i < n; i++) {
                double alpha = matrix[i][p] / matrix[p][p];
                matrix[i][p] = 0.0;
                for (int j = p + 1; j <= n; j++) {
                    matrix[i][j] -= alpha * matrix[p][j];
                }
            }
        }

        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double sum = 0.0;
            for (int j = i + 1; j < n; j++) {
                sum += matrix[i][j] * x[j];
            }
            x[i] = (matrix[i][n] - sum) / matrix[i][i];
        }
        return x;
    }

    public double predict(double x) {
        double y = 0;
        for (int i = 0; i < coefficients.length; i++) {
            y += coefficients[i] * Math.pow(x, i);
        }
        return y;
    }
    
    public void reset() {
        coefficients = new double[DEGREE + 1];
        xData.clear();
        yData.clear();
    }
    
    
    public void comparePredictions(int start, int end) {
        System.out.println("\nComparing Predictions for range x=" + start + " to x=" + end + ":");
        double totalError = 0.0;
        int count = 0;
        for (int xValue = start; xValue <= end; xValue++) {
            double predictedY = predict(xValue);
            double actualY = function(xValue);
            double error = (predictedY - actualY) / actualY;
            System.out.println("x=" + xValue + ", Predicted y=" + predictedY + ", Actual y=" + actualY + ", Relative Error=" + error);
            totalError += Math.abs(error);
            count++;
        }
        double averageError = totalError / count;
        System.out.println("Average Relative Error for range x=" + start + " to x=" + end + ": " + averageError);
    }


    public static void main(String[] args) {
        PolynomialRegression model = new PolynomialRegression();

        // Print out training data
        System.out.println("Training data with noise:");
        for (int i = 1; i <= N; i++) {
            double x = i;
            double y = model.function(x);
            System.out.println("x=" + x + ", y=" + y);
        }
        System.out.println("--------------------------");

        // Train the model
        model.train();

        // Test
        double xTest = 160;
        System.out.println("Predicted y for x=" + xTest + " is: " + model.predict(xTest));

        // Compare predictions
        model.comparePredictions(1100, 1300);  // Predict and compare for x values from 110 to 120
    }
    
}
