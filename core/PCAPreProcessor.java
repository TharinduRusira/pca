package core;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import com.google.common.base.Preconditions;

/**
 * Data preprocessor class for Principal Component Analysis This class has methods for mean normalization and feature
 * scaling References: PCA class notes, Andrew Ng, http://cs229.stanford.edu/notes/cs229-notes10.pdf A TUTORIAL ON
 * PRINCIPAL COMPONENT ANALYSIS Derivation, Discussion and Singular Value Decomposition
 * http://www.cs.princeton.edu/picasso/mats/PCA-Tutorial-Intuition_jp.pdf
 */
public class PCAPreProcessor {
  /**
   * Perform mean normalization on a given data matrix and return the normalized DenseMatrix
   * 
   * @param mat
   *          DenseVector with input data
   * 
   * @return normalized DenseVector
   */
  public DenseMatrix zeroMeanNormalize(DenseMatrix mat) {
    Preconditions.checkArgument(mat != null, "Argument cannot be null");
    int inputSize = mat.rowSize();
    DenseVector mu = new DenseVector(mat.columnSize());
    DenseMatrix normalized = new DenseMatrix(mat.rowSize(), mat.columnSize());
    mu.assign(0);
    // calculate the mean vector mu
    for (int i = 0; i < inputSize; i++) {
      mu.assign(mu.plus(mat.viewRow(i)));
    }
    mu.assign(mu.divide((double) inputSize));
    // minus mu from each data record
    for (int i = 0; i < inputSize; i++) {
      normalized.assignRow(i, mat.viewRow(i).minus(mu));
    }
    return normalized;
  }
  
  /**
   * Perform feature scaling on a given data matrix and return the scaled DenseMatrix
   * 
   * @param mat
   *          DenseVector with input data
   * @return scaled DensVector with scaled values
   */
  public DenseMatrix unitVarianceFeatureScale(DenseMatrix mat) {
    Preconditions.checkArgument(mat != null, "Argument cannot be null");
    int inputSize = mat.rowSize();
    int features = mat.columnSize();
    DenseMatrix scaled = new DenseMatrix(inputSize, features);
    DenseVector sigma2 = new DenseVector(features);
    sigma2.assign(0);
    for (int i = 0; i < inputSize; i++) {
      sigma2.assign(sigma2.plus(square((DenseVector) mat.viewRow(i))));
    }
    sigma2.assign(sigma2.divide(inputSize));
    
    for (int i = 0; i < features; i++) {
      scaled.assignColumn(i, mat.viewColumn(i).divide(Math.sqrt(sigma2.get(i))));
    }
    return scaled;
  }
  
  /**
   * 
   * Take a DenseVector as the argument and return a Vector with squared values of the argument's elements
   * 
   * @param v
   *          DenseVector with input vector
   * @return squared DenseVector with squared values
   */
  public DenseVector square(DenseVector v) {
    Preconditions.checkArgument(v != null, "Argument cannot be null");
    Preconditions.checkArgument(v.size() > 0, "Argument vector size should be positive");
    DenseVector squared = new DenseVector(v.size());
    for (int i = 0; i < squared.size(); i++) {
      double vi = v.get(i);
      squared.set(i, vi * vi);
    }
    return squared;
  }
}
