package edu.cwru.bayesmultinomialtestingbitwise.lattices;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import com.google.common.util.concurrent.AtomicDouble;

public abstract class ProductLatticeBitwiseBase implements ProductLatticeBitwise {
	protected int atoms;
	protected int variants;
	protected double[] posteriorProbabilities;
	protected int testCount = 0;
	protected int[] classificationStat;

	public ProductLatticeBitwiseBase() {
	}

	/**
	 * Constructor
	 * 
	 * @param atoms
	 * @param pi0
	 */
	public ProductLatticeBitwiseBase(int atoms, int variants, double[] pi0) {
		setAtoms(atoms);
		setVariants(variants);
		this.posteriorProbabilities = generatePriorProbabilities(pi0);
		classificationStat = new int[nominalPoolSize()];
	}

	/**
	 * Phatom constructor
	 * 
	 * @param atoms
	 * @param pi0
	 * @param i
	 */
	public ProductLatticeBitwiseBase(int atoms, int variants, int flag) {
		setAtoms(atoms);
		setVariants(variants);
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
	public ProductLatticeBitwiseBase(ProductLatticeBitwise model, int i) {
		setTestCount(model.getTestCount());
		setAtoms(model.getAtoms());
		setVariants(model.getVariants());
		if (i == 0) { // essential copy for broadcast variables
			copyClassificationStat(model.getClassificationStat());
			setPosteriorProbabilities(model.getPosteriorProbabilities());
		} else if (i == 1) { // simulation
			setPosteriorProbabilities(model.getPosteriorProbabilities());
			copyClassificationStat(model.getClassificationStat());

		} else if (i == 2) { // tree internal copy
			setClassificationStat(null);
		}
	}

	public int getAtoms() {
		return this.atoms;
	}

	public void setAtoms(int atoms) {
		this.atoms = atoms;
	}

	public int getVariants() {
		return this.variants;
	}

	public void setVariants(int variants) {
		this.variants = variants;
	}

	public int[] getClassificationStat() {
		return this.classificationStat;
	}

	public void setClassificationStat(int[] classificationStat) {
		this.classificationStat = classificationStat;
	}

	public void copyClassificationStat(int[] classificationStat) {
		this.classificationStat = classificationStat.clone();
	}

	public double[] getPosteriorProbabilities() {
		return this.posteriorProbabilities;
	}

	public void setPosteriorProbabilities(double[] posteriorProbabilities) {
		this.posteriorProbabilities = posteriorProbabilities;
	}

	public void copyPosteriorProbabilities(double[] posteriorProbabilities) {
		this.posteriorProbabilities = posteriorProbabilities.clone();
	}

	public int getTestCount() {
		return this.testCount;
	}

	public void setTestCount(int testCount) {
		this.testCount = testCount;
	}

	public int totalStates() {
		return 1 << (atoms * variants);
	}

	public int nominalPoolSize() {
		return atoms * variants;
	}

	public int[] getUpSet(int state) {
		int[] addIndex = new int[nominalPoolSize() - Long.bitCount(state)];
		int counter = 0, i, index;
		for (i = 0; i < nominalPoolSize(); i++) {
			index = (1 << i);
			if ((state & index) == 0)
				addIndex[counter++] = index;
		}
		return generatePowerSetAdder(addIndex, state);

	}

	/**
	 * helper function for get up set
	 * 
	 * @param addIndex
	 * @param state
	 * @return
	 */
	private static int[] generatePowerSetAdder(int[] addIndex, int state) {
		int n = addIndex.length, pow_set_size = 1 << n;
		int[] ret = new int[pow_set_size];
		int i, j, temp;
		for (i = 0; i < pow_set_size; i++) {
			temp = state;
			for (j = 0; j < n; j++) {
				/*
				 * Check if j-th bit in the counter is set If set then print j-th element from
				 * set
				 */
				if ((i & (1 << j)) > 0) {
					temp += addIndex[j];
				}
			}
			ret[i] = temp;
		}
		return ret;
	}

	public double[] generatePriorProbabilities(double[] pi0) {
		double[] ret = new double[totalStates()];
		for (int i = 0; i < totalStates(); i++) {
			ret[i] = generatePriorProbability(i, pi0);
		}
		return ret;
	}

	public double generatePriorProbability(int state, double[] pi0) {
		double prob = 1.0;
		for (int i = 0; i < nominalPoolSize(); i++) {
			if ((state & (1 << i)) == 0)
				prob *= pi0[i];
			else
				prob *= (1.0 - pi0[i]);
		}
		return prob;
	}

	public void resetTestCount() {
		this.testCount = 0;
	}

	public void updatePosteriorProbabilities(int experiment, int response, double upsetThresholdUp,
			double upsetThresholdLo) {
		posteriorProbabilities = calculatePosteriorProbabilities(experiment, response);
		updateClassifiedAtomsAndClassifiedState(upsetThresholdUp, upsetThresholdLo);
		testCount++;
	}

	public void updatePosteriorProbabilitiesInPlace(int experiment, int response, double upsetThresholdUp,
			double upsetThresholdLo) {
		calculatePosteriorProbabilitiesInPlace(experiment, response);
		updateClassifiedAtomsAndClassifiedState(upsetThresholdUp, upsetThresholdLo);
		testCount++;
	}

	public abstract double[] calculatePosteriorProbabilities(int experiment, int response);

	public abstract void calculatePosteriorProbabilitiesInPlace(int experiment, int response);

	public void updateClassifiedAtomsAndClassifiedState(double upsetThresholdUp, double upsetThresholdLo) {

		for (int i = 0; i < nominalPoolSize(); i++) {
			if (classificationStat[i] != 0)
				continue; // skip checking since it's already classified as either positive or negative
			int atom = 1 << i;
			double probMass = getUpSetProbabilityMass(atom);

			if (probMass < upsetThresholdLo)
				classificationStat[i] = -1; // classified as positive
			else if (probMass > (1 - upsetThresholdUp))
				classificationStat[i] = 1; // classified as negative
		}
	}

	public double getUpSetProbabilityMass(int state) {
		double upProbability = 0.0;
		for (int i : getUpSet(state))
			upProbability += posteriorProbabilities[i];
		return upProbability;
	}

	public boolean isClassified() {
		for (int i = 0; i < nominalPoolSize(); i++)
			if (classificationStat[i] == 0)
				return false;
		return true;
	}

	public int findHalvingStates(double prob) {
		int candidate = 0;
		int stateIter;
		int experiment;
		boolean isComplement = false;
		double min = Double.MAX_VALUE;
		int partitionID = 0;
		double[] partitionProbMass = new double[(1 << variants)];
		for (experiment = 0; experiment < (1 << atoms); experiment++) {
			// reset partitionProbMass
			for (int i = 0; i < (1 << variants); i++)
				partitionProbMass[i] = 0.0;
			// tricky: for each state, check each variant of actively
			// pooled subjects to see whether they are all 1.
			for (stateIter = 0; stateIter < totalStates(); stateIter++) {
				for (int variant = 0; variant < variants; variant++) {
					for (int l = 0; l < atoms; l++) {
						if ((experiment & (1 << l)) != 0 && (stateIter & (1 << (l * variants + variant))) == 0) {
							isComplement = true;
							break;
						}
					}
					partitionID |= (isComplement ? 0 : (1 << variant));
					isComplement = false; // reset flag
				}
				partitionProbMass[partitionID] += posteriorProbabilities[stateIter];
				partitionID = 0;
			}

			// for (int i = 0; i < totalStates(); i++) {
			// System.out.print(partitionMap[i] + " ");
			// }
			// System.out.println();

			double temp = 0.0;
			for (double d : partitionProbMass) {
				temp += Math.abs(d - prob);
			}
			if (temp < min) {
				min = temp;
				candidate = experiment;
			}

		}

		return candidate;

	}

	/**
	 * Multithreading halving algorithm for local acceleration
	 * 
	 * @param prob
	 * @return
	 */
	public int findHalvingStatesParallel(double prob) {
		AtomicInteger candidate = new AtomicInteger(0);
		AtomicDouble min = new AtomicDouble(Double.MAX_VALUE);
		IntStream.range(0, 1 << atoms).parallel().forEach(experiment -> {
			int partitionID = 0;
			boolean isComplement = false;
			double[] partitionProbMass = new double[(1 << variants)];
			// tricky: for each state, check each variant of actively
			// pooled subjects to see whether they are all 1.
			for (int stateIter = 0; stateIter < totalStates(); stateIter++) {
				partitionID = 0;
				for (int variant = 0; variant < variants; variant++) {
					for (int l = 0; l < atoms; l++) {
						if ((experiment & (1 << l)) != 0 && (stateIter & (1 << (l * variants + variant))) == 0) {
							isComplement = true;
							break;
						}
					}
					partitionID |= (isComplement ? 0 : (1 << variant));
					isComplement = false; // reset flag
				}
				partitionProbMass[partitionID] += posteriorProbabilities[stateIter];
			}

			double temp = 0.0;
			for (double d : partitionProbMass) {
				temp += Math.abs(d - prob);
			}
			if (temp < min.get()) {
				min.set(temp);
				candidate.set(experiment);
			}

		});

		return candidate.get();

	}

	public abstract double responseProbability(int experiment, int response, int trueState);
}
