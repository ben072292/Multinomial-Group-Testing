package edu.cwru.bayesmultinomialtestingbitwise.lattices;

import java.util.Arrays;
import java.util.stream.IntStream;

import edu.cwru.bayesmultinomialtestingbitwise.tool.Debug;

public class ProductLatticeBitwiseNonDilution extends ProductLatticeBitwiseBase {

	public ProductLatticeBitwiseNonDilution() {
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
	public ProductLatticeBitwiseNonDilution(int atoms, int variants, double[] pi0) {
		super(atoms, variants, pi0);
	}

	/**
	 * Phatom constructor
	 * 
	 * @param atoms
	 * @param pi0
	 * @param i
	 */
	public ProductLatticeBitwiseNonDilution(int atoms, int variants, int i) {
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
	public ProductLatticeBitwiseNonDilution(ProductLatticeBitwise model, int i) {
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
		double[] ret = new double[totalStates()];
		int partition = 0;

		// borrowed from find halving state function
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		int stateIter;
		double denominator = 0.0;
		boolean isComplement = false;
		for (stateIter = 0; stateIter < totalStates(); stateIter++) {
			ret[stateIter] = posteriorProbabilities[stateIter];
			partition = 0;
			for (int variant = 0; variant < variants; variant++) {
				isComplement = false;
				for (int l = 0; l < atoms; l++) {
					if ((experiment & (1 << l)) != 0 && (stateIter & 1 << (l * variants + variant)) == 0) {
						isComplement = true;
						break;
					}
				}
				partition |= (isComplement ? 0 : (1 << variant));
			}

			for(int i = 0; i < Long.bitCount(partition ^ response); i++){
				ret[stateIter] *= 0.005;
			}
			for(int i = 0; i < variants - Long.bitCount(partition ^ response); i++){
				ret[stateIter] *= 0.985;
			}
			denominator += ret[stateIter];
		}
		for (int i = 0; i < totalStates(); i++) {
			ret[i] /= denominator;
		}
		return ret;
	}

	public double[] calculatePosteriorProbabilitiesParallel(int experiment, int response) {
		double[] ret = new double[totalStates()];

		// borrowed from find halving state function
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		IntStream.range(0, totalStates()).parallel().forEach(stateIter -> {
			ret[stateIter] = posteriorProbabilities[stateIter];
			int partition = 0;
			boolean isComplement = false;
			for (int variant = 0; variant < variants; variant++) {
				isComplement = false;
				for (int l = 0; l < atoms; l++) {
					if ((experiment & (1 << l)) != 0 && (stateIter & 1 << (l * variants + variant)) == 0) {
						isComplement = true;
						break;
					}
				}
				partition |= (isComplement ? 0 : (1 << variant));
			}

			for(int i = 0; i < Long.bitCount(partition ^ response); i++){
				ret[stateIter] *= 0.005;
			}
			for(int i = 0; i < variants - Long.bitCount(partition ^ response); i++){
				ret[stateIter] *= 0.985;
			}
		});

		double denominator = Arrays.stream(ret).parallel().reduce(0.0, Double::sum);

		IntStream.range(0, totalStates()).parallel().forEach(stateIter -> {
			ret[stateIter] /= denominator;
		});
		
		return ret;
	}

	public void calculatePosteriorProbabilitiesInPlace(int experiment, int response) {
		int partition = 0;

		// borrowed from find halving state function
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		int stateIter;
		boolean isComplement = false;
		double denominator = 0.0;
		for (stateIter = 0; stateIter < totalStates(); stateIter++) {
			isComplement = false;
			partition = 0;
			for (int variant = 0; variant < variants; variant++) {
				for (int l = 0; l < atoms; l++) {
					if ((experiment & (1 << l)) != 0 && (stateIter & 1 << (l * variants + variant)) == 0) {
						isComplement = true;
						break;
					}
				}
				partition |= (isComplement ? 0 : (1 << variant));
				isComplement = false; // reset flag
			}

			for(int i = 0; i < Long.bitCount(partition ^ response); i++){
				posteriorProbabilities[stateIter] *= 0.005;
			}
			for(int i = 0; i < variants - Long.bitCount(partition ^ response); i++){
				posteriorProbabilities[stateIter] *= 0.985;
			}
			denominator += posteriorProbabilities[stateIter];
		}
		for (int i = 0; i < totalStates(); i++) {
			posteriorProbabilities[i] /= denominator;
		}
	}

	@Override
	public double responseProbability(int experiment, int response, int trueState, double[][] dilutionMatrix) {
		double ret = 1.0;
		int trueStateResponse = 0;
		int subjectCount = Long.bitCount(experiment);

		for (int i = 0; i < variants; i++) {
			int temp = 0;
			for (int j = 0; j < atoms; j++) {
				if ((experiment & (1 << j)) == 0)
					continue;
				if ((trueState & (1 << (j * variants + i))) != 0)
					temp++;
			}
			if (temp == subjectCount)
				trueStateResponse += (1 << i);
		}

		int error = Long.bitCount(trueStateResponse ^ response);
		int correct = variants - error;
		
		for(int i = 0; i < error; i++) ret *= 0.005;
		for(int i = 0; i < correct; i++) ret *= 0.985;
		return ret;
		

		
	}

	public static void main(String[] args) {
		int atoms = 2;
		int variants = 2;
		double[] pi0 = new double[atoms * variants];
		for (int i = 0; i < atoms * variants; i++) {
			pi0[i] = 0.02;
		}
		ProductLatticeBitwiseBase p = new ProductLatticeBitwiseNonDilution(atoms, variants, pi0);
		System.out.println(p.getUpSetProbabilityMass(1));
		Debug.showArray(p.posteriorProbabilities);
		Debug.showArray(p.getClassificationStat());
		System.out.println(p.isClassified());
		p.updatePosteriorProbabilitiesInPlace(3, 3, 0.01, 0.01);
		p.updatePosteriorProbabilitiesInPlace(3, 3, 0.01, 0.01);
		p.updatePosteriorProbabilitiesInPlace(3, 3, 0.01, 0.01);
		System.out.println();
		Debug.showArray(p.posteriorProbabilities);
		Debug.showArray(p.getClassificationStat());
		System.out.println(p.isClassified());

	}
}
