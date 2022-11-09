package edu.cwru.bayesmultinomialtestingbitwise.lattices;

import org.apache.commons.lang3.NotImplementedException;


public class ProductLatticeBitwiseDilution extends ProductLatticeBitwiseBase {

	public ProductLatticeBitwiseDilution() {
		super();
	}

	/**
	 * Default constructor serves as generating all required information of a
	 * lattice
	 * model. It's a costly process to construct a lattice instance but can greatly
	 * benefit many computation-intensive member methods.
	 * 
	 * @param atoms
	 * @param pi0
	 */
	public ProductLatticeBitwiseDilution(int atoms, int variants, double[] pi0) {
		super(atoms, variants, pi0);
	}

	/**
	 * Phatom constructor
	 * 
	 * @param atoms
	 * @param pi0
	 * @param i
	 */
	public ProductLatticeBitwiseDilution(int atoms, int variants, int i) {
		super(atoms, variants, i);
	}

	/**
	 * Copy constructor with different options used in different scenarios.
	 * i = 0: Used for Spark broadcast variables that gets shared with all computing
	 * nodes. i = 1: Used for constructing simulation trees in IS. i = 2: Used for
	 * constructing simulation trees in RS, KS.
	 * 
	 * @param model
	 * @param i
	 */
	public ProductLatticeBitwiseDilution(ProductLatticeBitwise model, int i) {
		super(model, i);
	}

	/**
	 * Calculate the posterior probability for each lattice.
	 * 
	 * @param S
	 * @param i = 1 : current experiment S are all negative
	 * @param i = 0 : current experiment S contains at least one positive case.
	 * @return Hash map contains all posterior probability for each lattice
	 */
	@Override
	public double[] calculatePosteriorProbabilities(int experiment, int response) {
        final double[][] dilutionMatrix = generateDilutionMatrix(0.99, 0.005);
		double[] ret = new double[totalStates()];
		int stateIter;
		double denominator = 0.0;
		for (stateIter = 0; stateIter < totalStates(); stateIter++) {
            ret[stateIter] = posteriorProbabilities[stateIter] * responseProbability(experiment, response, stateIter, dilutionMatrix);
			denominator += ret[stateIter];
		}
		for (int i = 0; i < totalStates(); i++) {
			ret[i] /= denominator;
		}
		return ret;
	}

	public double[] calculatePosteriorProbabilitiesParallel(int experiment, int response) {
		throw new NotImplementedException();
	}

	public void calculatePosteriorProbabilitiesInPlace(int experiment, int response) {
		throw new NotImplementedException();
	}

	@Override
	public double responseProbability(int experiment, int response, int trueState, final double[][] dilutionMatrix) {
        double ret = 1.0;
		int trueStatePerVariant = 0;
		int experimentLength = Long.bitCount(experiment);
        for (int variant = 0; variant < variants; variant++) {
			trueStatePerVariant = 0;
            for (int l = 0; l < atoms; l++)
				trueStatePerVariant += (trueState & (1 << (l * variants + variant))) != 0 
				? (1 << l) 
				: 0;
            ret *= (response & (1 << variant)) != 0
				? dilutionMatrix[experimentLength-1][experimentLength-Long.bitCount(experiment & trueStatePerVariant)] 
				: 1.0 - dilutionMatrix[experimentLength-1][experimentLength-Long.bitCount(experiment & trueStatePerVariant)];
           
        }
        return ret;
	}

	public static void main(String[] args) {
		double[] pi0 = {0.02, 0.02, 0.02, 0.02, 0.02, 0.02};
		ProductLatticeBitwiseDilution p = new ProductLatticeBitwiseDilution(3, 2, pi0);
		System.out.println(p.responseProbability(7, 3, 63, p.generateDilutionMatrix(0.99, 0.005)));

	}
}
