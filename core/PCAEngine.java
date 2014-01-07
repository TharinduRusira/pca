package core;

import java.util.HashMap;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.SingularValueDecomposition;

import com.google.common.base.Preconditions;

import misc.PCAHelperFunctions;

public class PCAEngine implements AbstractPCAEngine {
  DenseMatrix input, reducedMatrix;
  int inputSize;
  int features;
  int reducedSize;
  
  public PCAEngine(DenseMatrix in) {
    this.input = in;
    this.inputSize = in.rowSize();
    this.features = in.columnSize();
    // default value is set to 1 which gives the principal component
    this.reducedSize = 1;
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
   * @return {@link DenseMatrix} matrix of k-principal components
   */
  @Override
  public DenseMatrix runPCA() {
    reducedMatrix = new DenseMatrix(inputSize, reducedSize);
    // implement pca here
    return reducedMatrix;
  }
  
  /**
   * 
   * @return {@link DenseMatrix} co-variance matrix of the input matrix
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
   * A vectorized(matrix multiplication) implementation of covariance matrix
   * 
   * @return {@link DenseMatrix} covariance matrix of the input matrix
   * 
   */
  public DenseMatrix getCovarianceMatrixVectorized() {
    DenseMatrix covar = new DenseMatrix(features, features);
    covar = (DenseMatrix) input.transpose().times(input);
    covar = (DenseMatrix) covar.divide(inputSize);
    return covar;
  }
  
  /**
   * Calculates Eigen Vectors of a given Co-variance matrix using Single Value Decomposition(SVD) method Mahout math
   * library does not provide a way to calculate Eigenvalues directly. Hence, principal component analysis is done with
   * SVD. For a matrix with real elements, Eigenvalues = squares of singular values
   * 
   * @return {@link DenseVector} eigenvalues
   */
  @Override
  public DenseVector getEigenValues() {
    SingularValueDecomposition svd = new SingularValueDecomposition(input);
    double[] singularValues = svd.getSingularValues();
    Preconditions.checkNotNull(singularValues, "Singular Values returned null");
    Preconditions.checkArgument(singularValues.length != 0, "No singular values found for the input matrix");
    DenseVector eigenV = PCAHelperFunctions.square(new DenseVector(singularValues));
    return eigenV;
  }
  
  /**
   * Right singular vectors are equivalent to eigenvectors. SingleValueDecomposition class will be used to obtain right
   * singular values of the input matrix directly
   * 
   * @param in
   *          {@link DenseMatrix}
   * @return sorted eigenvectors as columns of a {@link DenseMatrix}
   */
  public DenseMatrix getEigenVectors(DenseMatrix in) {
    SingularValueDecomposition svd = new SingularValueDecomposition(in);
    DenseMatrix eigenVec = (DenseMatrix) svd.getV();
    return eigenVec;
  }
  
  /**
   * 
   * @return a Hashmap containing eigenvalues and corresponding eigenvectors
   */
  public HashMap<Double,DenseVector> mapEigenvaluesToEigenVectors(double[] singularValues, Matrix eigenVectors) {
    HashMap<Double,DenseVector> eigenMap = new HashMap<>(singularValues.length);
    for (int i = 0; i < singularValues.length; i++) {
      // eigenvalue = singular value ^2
      // eigenvector = right singular vector
      eigenMap.put(singularValues[i] * singularValues[i], (DenseVector) eigenVectors.viewRow(i));
    }
    return eigenMap;
  }
  
  /**
   * Return k eigenvectors corresponding to top k eigenvalues
   * 
   * @param v
   *          {@link DenseMatrix}
   * @return topk {@link DenseMatrix}
   */
  public DenseMatrix getKPrincipalComponents(DenseMatrix v) {
    // write code
    return null;
  }
  
}
