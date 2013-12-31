package tests;

import static org.junit.Assert.*;
import org.apache.mahout.math.DenseMatrix;
import org.junit.Before;
import org.junit.Test;

import core.PCAEngine;

public class PCAEngineTest {
  
  
  @Test
  public void covarianceMatrixTest() {
    DenseMatrix input = new DenseMatrix(new double[][] { {1, 2, 3}, {4, 5, 6}});
    DenseMatrix expected = new DenseMatrix(new double[][] { {8.5, 11.0, 13.5}, {11.0, 14.5, 18.0}, {13.5, 18.0, 22.5}});
    DenseMatrix result = new PCAEngine(input).getCovarianceMatrix();
    assertEquals("Covariance Matrix Test failed", expected.asFormatString(), result.asFormatString());
  }
  @Test
  public void getEigenVectorsTest(){
    
  }
  
}
