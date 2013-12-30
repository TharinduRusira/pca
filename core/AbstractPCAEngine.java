package core;

import org.apache.mahout.math.DenseMatrix;
/** 
 * 
 * Input data matrix is supposed to be of the following format.
 * Rows --> Data records
 * Columns --> features 
 *  
 */

public interface AbstractPCAEngine {
  public DenseMatrix runPCA();
  public DenseMatrix getCovarianceMatrix();
  public DenseMatrix getEigenVectors();
  
}
