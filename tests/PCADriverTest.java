package tests;

import static org.junit.Assert.*;

import org.junit.Test;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;

import core.PCADriver;
import core.PCAPreProcessor;

public class PCADriverTest extends PCADriver {
  
  @Test
  public void meanNormalizeTest() {
    DenseMatrix input = new DenseMatrix(new double[][] { {1, 2, 3}, {100, 500, 2}, {300, 6, 40}});
    DenseMatrix expected = new DenseMatrix(new double[][] {
        {-132.66666666666666, -167.33333333333334, -12.0},
        {-33.66666666666666, 330.66666666666663, -13.0},
        {166.33333333333334, -163.33333333333334, 25.0}});
    PCAPreProcessor prepro = new PCAPreProcessor();
    DenseMatrix result = prepro.meanNormalize(input);
    assertEquals("meanNormalizeTest failed", expected.asFormatString(), result.asFormatString());
  }
  
  @Test
  public void featureScaleTest() {
    DenseMatrix input = new DenseMatrix(new double[][] { {1, 2, 3}, {4, 5, 6}});
    DenseMatrix expected = new DenseMatrix(new double[][] {
        {0.34299717028501764, 0.5252257314388902, 0.6324555320336759},
        {1.3719886811400706, 1.3130643285972254, 1.2649110640673518}});
    PCAPreProcessor prepro = new PCAPreProcessor();
    DenseMatrix result = prepro.featureScale(input);
    assertEquals("featureScaleTest failed", expected.asFormatString(), result.asFormatString());
    
  }
  
  @Test
  public void vectorSquareTest() {
    DenseVector input = new DenseVector(new double[] {1, 2, 3});
    PCAPreProcessor prepro = new PCAPreProcessor();
    DenseVector result = prepro.square(input);
    DenseVector expected = new DenseVector(new double[] {1, 4, 9});
    assertEquals("vectorSquareTest failed", expected, result);
    
  }
  
}
