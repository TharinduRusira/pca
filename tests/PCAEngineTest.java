package tests;

import static org.junit.Assert.*;

import org.apache.mahout.math.DenseMatrix;
import org.junit.Test;

import core.PCAEngine;

public class PCAEngineTest {

	@Test
	public void covarianceMatrixTest() {
		DenseMatrix input = new DenseMatrix(new double[][]{{1,2,3},{4,5,6}});
		PCAEngine pcae = new PCAEngine(input, 3);
		System.out.println(pcae.getCovarianceMatrix());
		//assertEquals("Covariance Matrix Test failed",0, 1);
	}

}
