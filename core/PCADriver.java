package core;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;

public class PCADriver {
  
  private int inputSize = -1;
  private PCAPreProcessor prepro;
  
  public void run(DenseMatrix input, boolean doMeanNormalize, boolean doFeatureScale) {
    this.inputSize = input.rowSize();
    if (doMeanNormalize || doFeatureScale) {
      prepro = new PCAPreProcessor();
    }
    if (doMeanNormalize) {
      input = prepro.zeroMeanNormalize(input);
    }
    if (doFeatureScale) {
      input = prepro.unitVarianceFeatureScale(input);
    }
    
  }
  
}
