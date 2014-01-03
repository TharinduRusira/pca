package misc;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixView;

import com.google.common.base.Preconditions;

/**
 * This class implements methods required to miscellaneous activities that are not directly provided by Mahout
 * libraries.
 */
public class PCAHelperFunctions {
  
  public DenseMatrix matrixViewToDenseMatrix(MatrixView m) {
    
    return new DenseMatrix(new double[][] {});
  }
  
  /**
   * 
   * Take a DenseVector as the argument and return a Vector with squared values of the argument's elements
   * 
   * @param v
   *          DenseVector with input vector
   * @return squared DenseVector with squared values
   */
  public static DenseVector square(DenseVector v) {
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
