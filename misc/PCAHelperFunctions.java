package misc;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.MatrixView;

/**
 * This class implements methods required to miscellaneous activities that are not directly provided by Mahout
 * libraries.
 */
public class PCAHelperFunctions {
  
  public DenseMatrix matrixViewToDenseMatrix(MatrixView m) {
    
    return new DenseMatrix(new double[][] {});
  }
  
}
