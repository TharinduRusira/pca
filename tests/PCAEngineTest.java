package tests;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.hamcrest.core.IsInstanceOf;
import org.junit.Before;
import org.junit.Test;

import core.PCAEngine;

public class PCAEngineTest {
  
  @Test
  public void testPCAEngineConstructor2(){
    DenseMatrix input = new DenseMatrix(new double[][] { {1, 2, 3}, {4, 5, 6}});
    try{
      // this call should generate an IllegalArgumentException
      new PCAEngine(input, 5);
      assertTrue("checkPCAEngineConstructor2 failed", false);
    }catch(IllegalArgumentException e){
      assertTrue(true);
    }catch(Exception other){
      assertTrue("checkPCAEngineConstructor2 failed",false);
    }
  }
  
  @Test
  public void testCovarianceMatrix() {
    DenseMatrix input = new DenseMatrix(new double[][] { {1, 2, 3}, {4, 5, 6}});
    DenseMatrix expected = new DenseMatrix(new double[][] { {8.5, 11.0, 13.5}, {11.0, 14.5, 18.0}, {13.5, 18.0, 22.5}});
    DenseMatrix result = new PCAEngine(input).getCovarianceMatrix();
    assertEquals("Covariance Matrix Test failed", expected.asFormatString(), result.asFormatString());
  }
  @Test
  public void testCovarianceMatrixVectorized(){
    DenseMatrix input = new DenseMatrix(new double[][] { {1, 2, 3}, {4, 5, 6}});
    DenseMatrix expected = new DenseMatrix(new double[][] { {8.5, 11.0, 13.5}, {11.0, 14.5, 18.0}, {13.5, 18.0, 22.5}});
    DenseMatrix result = new PCAEngine(input).getCovarianceMatrixVectorized();
    assertEquals("Covariance Matrix Test failed", expected.asFormatString(), result.asFormatString());
  }

  @Test
  public void testGetEigenVectors(){
    
  }
  
}
