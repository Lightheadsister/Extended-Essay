import java.util.Arrays;

public class KFoldCV {

    private int k;
    public PolynomialRegression model;
    private double[] xs;
    private double[] ys;
    private final int DEGREE = 10;
    
    public KFoldCV(PolynomialRegression model, int dataSize) {
        this.k = dataSize / 10 ;
        if (this.k <= 1) {
            throw new IllegalArgumentException("Number of data points is too low for meaningful cross-validation.");
        }
        this.model = model;
        this.xs = new double[dataSize];
        this.ys = new double[dataSize];

        // Generating the dataset based on provided size
        for (int i = 0; i < dataSize; i++) {
            xs[i] = i + 1;
            ys[i] = function(xs[i]);
        }

        // Printing out the generated dataset
        System.out.println("Generated Training Data:");
        for (int i = 0; i < dataSize; i++) {
            System.out.println("x=" + xs[i] + ", y=" + ys[i]);
        }
        System.out.println("--------------------------");
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
   
    
    public void train1(double[] trainX, double[] trainY) {
        double[][] matrix = new double[DEGREE + 1][DEGREE + 2];

        for (int i = 0; i <= DEGREE; i++) {
            for (int j = 0; j <= DEGREE; j++) {
                matrix[i][j] = 0;
                for (int k = 0; k < trainX.length; k++) {
                    matrix[i][j] += Math.pow(trainX[k], i + j);
                }
            }
        }

        double[] Y = new double[DEGREE + 1];
        for (int i = 0; i <= DEGREE; i++) {
            Y[i] = 0;
            for (int j = 0; j < trainX.length; j++) {
                Y[i] += Math.pow(trainX[j], i) * trainY[j];
            }
        }

        for (int i = 0; i <= DEGREE; i++) {
            matrix[i][DEGREE + 1] = Y[i];
        }

        model.coefficients = gaussianElimination(matrix);
    }

    private double[] gaussianElimination(double[][] matrix) {
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
    
    public double validate() {
        int foldSize = xs.length / k;  
        double[] errors = new double[k];

        for (int fold = 0; fold < k; fold++) {
            int testIndex = (fold + 1) * foldSize - 1; // Last data point in the current fold

            // Extract training data for this fold (including data from previous folds)
            double[] trainX = new double[(fold + 1) * foldSize - 1]; // Excluding the last data point in the current fold
            double[] trainY = new double[(fold + 1) * foldSize - 1]; // Excluding the last data point in the current fold

            for (int i = 0; i < trainX.length; i++) {
                trainX[i] = xs[i];
                trainY[i] = ys[i];
            }

            train1(trainX, trainY);

            // Only test the last data point in the current fold
            double predicted = model.predict(xs[testIndex]);
            double error = (predicted - ys[testIndex]) / ys[testIndex];
            errors[fold] = error;

            System.out.println( error);
        }

        double avgError = Arrays.stream(errors).average().getAsDouble();
        System.out.println("---------------------------------------------------");
        System.out.println("Average Error after " + k + "-fold CV: " + avgError);
        return avgError;
    }

    public  static void main(String[] args) {
        PolynomialRegression model = new PolynomialRegression();
        int dataSize = 1000;  // You can adjust this value as needed
        KFoldCV cv = new KFoldCV(model, dataSize);
        cv.validate();
    }

    

} 