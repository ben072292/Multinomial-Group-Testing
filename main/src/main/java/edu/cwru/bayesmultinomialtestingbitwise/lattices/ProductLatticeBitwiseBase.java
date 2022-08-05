package edu.cwru.bayesmultinomialtestingbitwise.lattices;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import com.google.common.util.concurrent.AtomicDouble;

public abstract class ProductLatticeBitwiseBase {
	protected int atoms;
	protected int variants;
	protected double[] posteriorProbabilityMap; // keyset memory allocated by E
	protected int testCounter = 0;
	protected int[] classificationStat;

	public ProductLatticeBitwiseBase() {
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
	public ProductLatticeBitwiseBase(int atoms, int variants, double[] pi0) {
		this.atoms = atoms;
		this.variants = variants;
		this.posteriorProbabilityMap = generatePriorProbabilityMap(pi0);
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
		this.atoms = atoms;
		this.variants = variants;
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
	public ProductLatticeBitwiseBase(ProductLatticeBitwiseBase model, int i) {
		this.testCounter = model.testCounter;
		if (i == 0) { // essential copy for broadcast variables
			this.atoms = model.atoms;
			this.variants = model.variants;
			this.classificationStat = model.classificationStat.clone();
			this.posteriorProbabilityMap = model.posteriorProbabilityMap;
		} else if (i == 1) { // simulation
			this.atoms = model.atoms;
			this.variants = model.variants;
			this.posteriorProbabilityMap = model.posteriorProbabilityMap;
			this.classificationStat = model.classificationStat.clone();

		} else if (i == 2) { // tree internal copy
			this.atoms = model.atoms;
			this.variants = model.variants;
			this.classificationStat = null;
		}
	}

	public int getAtoms() {
		return this.atoms;
	}

	public int getVariants() {
		return this.variants;
	}

	public int[] getClassificationStat() {
		return this.classificationStat;
	}

	public double[] getPosteriorProbabilityMap() {
		return this.posteriorProbabilityMap;
	}

	public int getTestCounter() {
		return this.testCounter;
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

	public double[] generatePriorProbabilityMap(double[] pi0) {
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

	public void resetTestCounter() {
		this.testCounter = 0;
	}

	public abstract void updatePosteriorProbability(int experiment, int response, double upsetThresholdUp,
			double upsetThresholdLo);

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
			upProbability += posteriorProbabilityMap[i];
		return upProbability;
	}

	public boolean isClassified() {
		for (int i = 0; i < nominalPoolSize(); i++)
			if (classificationStat[i] == 0)
				return false;
		return true;
	}

	public int findHalvingState(double prob) {
		int candidate = 0;
		int stateIter;
		int experiment;
		boolean isComplement = false;
		double min = Double.MAX_VALUE;
		int[] partitionMap = new int[totalStates()];
		for (experiment = 0; experiment < (1 << atoms); experiment++) {
			for (stateIter = 0; stateIter < totalStates(); stateIter++) {
				partitionMap[stateIter] = 0;
			}

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
					partitionMap[stateIter] |= (isComplement ? 0 : (1 << variant));
					isComplement = false; // reset flag
				}
			}

			// for (int i = 0; i < totalStates(); i++) {
			// System.out.print(partitionMap[i] + " ");
			// }
			// System.out.println();

			double[] partitionProbMass = new double[(1 << variants)];
			for (stateIter = 0; stateIter < totalStates(); stateIter++) {
				partitionProbMass[partitionMap[stateIter]] += posteriorProbabilityMap[stateIter];
			}
			double temp = 0.0;
			for (double d : partitionProbMass) {
				temp += Math.abs(d - prob);
			}
			if (temp < min) {
				min = temp;
				candidate = experiment;
			}

			// reset partitionMap
			for (int i = 0; i < totalStates(); i++)
				partitionMap[i] = 0;

		}

		return candidate;

	}

	public int findHalvingStateParallel(double prob) {
		AtomicInteger candidate = new AtomicInteger(0);
		AtomicDouble min = new AtomicDouble(Double.MAX_VALUE);
		IntStream.range(0, 1 << atoms).parallel().forEach(experiment -> {
			int[] partitionMap = new int[totalStates()];
			boolean isComplement = false;
			for (int stateIter = 0; stateIter < totalStates(); stateIter++) {
				partitionMap[stateIter] = 0;
			}

			// tricky: for each state, check each variant of actively
			// pooled subjects to see whether they are all 1.
			for (int stateIter = 0; stateIter < totalStates(); stateIter++) {
				for (int variant = 0; variant < variants; variant++) {
					for (int l = 0; l < atoms; l++) {
						if ((experiment & (1 << l)) != 0 && (stateIter & (1 << (l * variants + variant))) == 0) {
							isComplement = true;
							break;
						}
					}
					partitionMap[stateIter] |= (isComplement ? 0 : (1 << variant));
					isComplement = false; // reset flag
				}
			}

			// for (int i = 0; i < totalStates(); i++) {
			// System.out.print(partitionMap[i] + " ");
			// }
			// System.out.println();

			double[] partitionProbMass = new double[(1 << variants)];
			for (int stateIter = 0; stateIter < totalStates(); stateIter++) {
				partitionProbMass[partitionMap[stateIter]] += posteriorProbabilityMap[stateIter];
			}
			double temp = 0.0;
			for (double d : partitionProbMass) {
				temp += Math.abs(d - prob);
			}
			if (temp < min.get()) {
				min.set(temp);
				candidate.set(experiment);
			}

			partitionMap = null;

		});

		return candidate.get();

	}

	public abstract double computeResponseProbabilityUsingTrueState(int experiment, int response, int trueState);
}
