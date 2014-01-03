package core;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import com.google.common.base.Preconditions;

public class PCAEngine implements AbstractPCAEngine {
  DenseMatrix input, reducedMatrix;
  int inputSize;
  int features;
  int reducedSize;
  
  public PCAEngine(DenseMatrix in) {
    this.input = in;
    this.inputSize = in.rowSize();
    this.features = in.columnSize();
    // default value is set as the same numbers as the number of original features
    this.reducedSize = features;
  }
  
  public PCAEngine(DenseMatrix in, int k) {
    this.input = in;
    this.inputSize = in.rowSize();
    this.features = in.columnSize();
    Preconditions.checkArgument(k <= features,
        "reduced dimension should be less than or equal to the number of original features");
    this.reducedSize = k;
  }
  
  /**
   * 
   * @return
   */
  @Override
  public DenseMatrix runPCA() {
    reducedMatrix = new DenseMatrix(inputSize, reducedSize);
    // implement pca here
    return reducedMatrix;
  }
  
  /**
   * 
   * @return
   */
  @Override
  public DenseMatrix getCovarianceMatrix() {
    DenseMatrix covar = new DenseMatrix(features, features);
    // calculate Covariance matrix
    Matrix rowi;
    for (int i = 0; i < this.inputSize; i++) {
      rowi = input.viewPart(i, 1, 0, features);
      covar.assign(covar.plus(rowi.transpose().times(rowi)));
    }
    covar.assign(covar.divide(inputSize));
    return covar;
  }
  
  /**
   * A vectorized implementation of covariance matrix
   * 
   */
  public DenseMatrix getCovarianceMatrixVectorized() {
    DenseMatrix covar = new DenseMatrix(features, features);
    covar = (DenseMatrix) input.transpose().times(input);
    covar = (DenseMatrix) covar.divide(inputSize);
    return covar;
  }
  
  /**
   * Calculates Eigen Vectors of a given Co-variance matrix using Single Value Decomposition(SVD) method
   * 
   * @return u
   */
  @Override
  public DenseMatrix getEigenVectors() {
    
    return null;
  }
}
