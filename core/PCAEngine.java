package core;

import org.apache.mahout.math.DenseMatrix;

public class PCAEngine implements AbstractPCAEngine{
  DenseMatrix input,reducedMatrix;
  int inputSize;
  int features;
  int reducedSize;
  
  public PCAEngine(DenseMatrix in){
	  this.input = in;
	  this.inputSize = in.rowSize();
	  this.features = in.columnSize();
	  this.reducedSize = features; //default values set as the same numbers of original features
  }
  
  public PCAEngine(DenseMatrix in, int k) {
    this.input = in;
    this.inputSize = in.rowSize();
    this.features = in.columnSize();
    this.reducedSize = k;
  }
  /**
   * 
   * @return
   */
  @Override
  public DenseMatrix runPCA(){
    reducedMatrix = new DenseMatrix(inputSize, reducedSize);
    // implement pca here
    return reducedMatrix;
  }
  /**
   * 
   * @return
   */
  @Override
  public DenseMatrix getCovarianceMatrix(){
    DenseMatrix covar = new DenseMatrix(inputSize, features);
    // calculate covariance matrix
    DenseMatrix rowi;
    for(int i=0;i<this.inputSize;i++){
    	//read data records line by line
    	rowi = (DenseMatrix)input.viewPart(i, 1,0,features);
    	covar.assign(covar.plus(rowi.transpose().times(rowi)));
    }
    covar.assign(covar.divide(inputSize));
    return covar;
  }
  /**
   * Implements Eigen Vectors using Single Value Decomposition(SVD) method 
   * @return
   */
  @Override
  public DenseMatrix getEigenVectors() {
    // TODO Auto-generated method stub
    return null;
  }
}
