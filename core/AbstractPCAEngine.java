package core;

import org.apache.mahout.math.DenseMatrix;

public interface AbstractPCAEngine {
  public DenseMatrix runPCA();
  public DenseMatrix getCovarianceMatrix();
  public DenseMatrix getEigenVectors();
  
}
