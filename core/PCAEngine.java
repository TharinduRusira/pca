package core;

import org.apache.mahout.math.DenseMatrix;

public class PCAEngine implements AbstractPCAEngine{
  DenseMatrix input,reducedMatrix;
  int inputSize;
  int features;
  int reducedSize;
  
  public PCAEngine(DenseMatrix in, int k) {
    this.input = in;
    this.inputSize = in.rowSize();
    this.features = in.columnSize();
    this.reducedSize = k;
  }
  /**
   * 
   * @return
   */
  @Override
  public DenseMatrix runPCA(){
    reducedMatrix = new DenseMatrix(inputSize, reducedSize);
    // implement pca here
    return reducedMatrix;
  }
  /**
   * 
   * @return
   */
  @Override
  public DenseMatrix getCovarianceMatrix(){
    DenseMatrix covar = new DenseMatrix(inputSize, features);
    // calculate covariance matrix
    return covar;
  }
  /**
   * Implements Eigen Vectors using Single Value Decomposition(SVD) method 
   * @return
   */
  @Override
  public DenseMatrix getEigenVectors() {
    // TODO Auto-generated method stub
    return null;
  }
}
