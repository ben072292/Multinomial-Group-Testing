package edu.cwru.bayesmultinomialtesting.lattices;

import java.util.ArrayList;

import edu.cwru.bayesmultinomialtesting.utilities.Pair;

public class ProductLatticeNonDilution extends ProductLatticeBase {

	public ArrayList<Pair<String, String>> experimentHistory = new ArrayList<>();

	public ProductLatticeNonDilution() {
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
	public ProductLatticeNonDilution(ArrayList<String> atoms, ArrayList<Double> pi0) {
		super(atoms, pi0);
	}

	/**
	 * Phatom constructor
	 * 
	 * @param atoms
	 * @param pi0
	 * @param i
	 */
	public ProductLatticeNonDilution(ArrayList<String> atoms, ArrayList<Double> pi0, int i) {
		super(atoms, pi0, i);
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
	public ProductLatticeNonDilution(ProductLatticeNonDilution model, int i) {
		super(model, i);
		this.experimentHistory = new ArrayList<>(model.experimentHistory);
	}

	@Override
	public void updatePosteriorProbability(String experimentInBit, String responsesInBit, double upsetThresholdUp,
			double upsetThresholdLo) {
		this.posteriorProbabilityMap = calculatePosteriorProbabilities(experimentInBit, responsesInBit);
		updateClassifiedAtomsAndClassifiedState(upsetThresholdUp, upsetThresholdLo);
		String subject = "";
		for (int i = 0; i < experimentInBit.length(); i++) {
			if (experimentInBit.charAt(i) == '1')
				subject += numberToChar(i);
		}
		experimentHistory.add(new Pair<String, String>(subject, responsesInBit));

		this.testCounter++;
	}

	private String numberToChar(int i) {
		return i > -1 && i < 26 ? String.valueOf((char) (i + 65)) : null;
	}

	/**
	 * Calculate the posterior probability for each lattice.
	 * 
	 * @param S
	 * @param i = 1 : current experiment S are all negative
	 * @param i = 0 : current experiment S contains at least one positive case.
	 * @return Hash map contains all posterior probability for each lattice
	 */
	public double[] calculatePosteriorProbabilities(String experimentInBit, String responsesInBit) {
		double[] ret = new double[totalStates];

		String[] partitionMap = new String[totalStates];
		for (int j = 0; j < totalStates; j++) {
			partitionMap[j] = ""; // otherwise initilized to null
		}

		// borrowed from find halving state function
		// tricky: for each state, check each variant of actively
		// pooled subjects to see whether they are all 1.
		for (int j = 0; j < totalStates; j++) {
			String stateBit = intToBitStates(j);
			for (int k = 0; k < variant; k++) {
				boolean isComplement = false;
				for (int l = 0; l < experimentInBit.length(); l++) {
					if (experimentInBit.charAt(l) == '1') {
						if (stateBit.charAt(l * variant + k) != '1') {
							isComplement = true;
							break;
						}
					}
				}
				partitionMap[j] += isComplement ? "0" : "1";
			}
		}

		double denominator = 0.0;
		for (int i = 0; i < totalStates; i++) {
			if (partitionMap[i].equals(responsesInBit))
				ret[i] = posteriorProbabilityMap[i] * 0.985;
			else
				ret[i] = posteriorProbabilityMap[i] * 0.005;
			denominator += ret[i];
		}

		for (int i = 0; i < totalStates; i++) {
			ret[i] /= denominator;
		}

		return ret;
	}

	@Override
	public double computeResponseProbabilityUsingTrueState(String experimentInBit, String responseInBit,
			String trueState) {
		String trueStateResponse = "";
		int subjectCount = 0;
		for (int i = 0; i < experimentInBit.length(); i++) {
			if (experimentInBit.charAt(i) == '1')
				subjectCount++;
		}

		for (int i = 0; i < variant; i++) {
			int temp = 0;
			for (int j = 0; j < atoms.size(); j++) {
				if (experimentInBit.charAt(j) == '0')
					continue;
				if (trueState.charAt(j * variant + i) == '1')
					temp++;
			}
			if (temp < subjectCount)
				trueStateResponse += "0";
			else if (temp == subjectCount)
				trueStateResponse += "1";
			else {
				System.err.println("Error!");
				System.exit(1);
			}
		}
		return trueStateResponse.equals(responseInBit) ? 0.985 : 0.005;
	}
}
