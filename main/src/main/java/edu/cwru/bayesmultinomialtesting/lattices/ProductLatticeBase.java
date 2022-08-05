package edu.cwru.bayesmultinomialtesting.lattices;

import edu.cwru.bayesmultinomialtesting.utilities.*;

import java.util.ArrayList;

public abstract class ProductLatticeBase {
	protected ArrayList<String> atoms;
	protected int variant;
	protected int totalStates;
	protected ArrayList<Double> pi0; // prior probability of population
	protected double[] posteriorProbabilityMap; // keyset memory allocated by E
	protected int testCounter = 0;
	protected ArrayList<Integer> classifiedAtoms = new ArrayList<>();
	protected ArrayList<Integer> classifiedAsPositive = new ArrayList<>();
	protected boolean classified = false;

	public ProductLatticeBase() {
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
	public ProductLatticeBase(ArrayList<String> atoms, ArrayList<Double> pi0) {
		this.atoms = atoms;
		this.atoms.sort(String::compareToIgnoreCase); // sort in alphabetic order
		this.variant = pi0.size() / atoms.size();
		this.totalStates = (1 << pi0.size());
		this.pi0 = pi0;
		this.posteriorProbabilityMap = generatePriorProbabilityMap();
	}

	/**
	 * Phatom constructor
	 * 
	 * @param atoms
	 * @param pi0
	 * @param i
	 */
	public ProductLatticeBase(ArrayList<String> atoms, ArrayList<Double> pi0, int i) {
		this.atoms = atoms;
		this.atoms.sort(String::compareToIgnoreCase); // sort in alphabetic order
		this.pi0 = pi0;
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
	public ProductLatticeBase(ProductLatticeBase model, int i) {
		this.testCounter = model.testCounter;
		if (i == 0) { // essential copy for broadcast variables
			this.atoms = model.atoms;
			this.pi0 = model.pi0;
			this.variant = model.variant;
			this.totalStates = model.totalStates;
			this.classifiedAtoms = new ArrayList<>(model.classifiedAtoms);
			this.classifiedAsPositive = new ArrayList<>(model.classifiedAsPositive);
			this.posteriorProbabilityMap = model.posteriorProbabilityMap;
			this.classified = model.classified;
		} else if (i == 1) { // simulation
			this.atoms = model.atoms;
			this.pi0 = model.pi0;
			this.variant = model.variant;
			this.totalStates = model.totalStates;
			this.posteriorProbabilityMap = model.posteriorProbabilityMap;
			this.classifiedAtoms = new ArrayList<>(model.classifiedAtoms);
			this.classifiedAsPositive = new ArrayList<>(model.classifiedAsPositive);
			this.classified = model.classified;

		} else if (i == 2) { // tree internal copy
			this.atoms = model.atoms;
			this.pi0 = model.pi0;
			this.variant = model.variant;
			this.totalStates = model.totalStates;
			this.classifiedAsPositive = model.classifiedAsPositive;
			this.classified = model.classified;
			this.classifiedAtoms = null;
		}
	}

	public int bitToInt(String bit) {
		return Integer.parseInt(bit, 2);
	}

	public String intToBitStates(int i) {
		String s = Integer.toBinaryString(i);
		int prefix = pi0.size() - s.length();
		for (int j = 0; j < prefix; j++) {
			s = "0" + s;
		}
		return s;

	}

	public String intToBitResponse(int i) {
		String s = Integer.toBinaryString(i);
		int prefix = atoms.size() - s.length();
		for (int j = 0; j < prefix; j++) {
			s = "0" + s;
		}
		return s;
	}

	public String intToBitExperiments(int i) {
		String s = Integer.toBinaryString(i);
		int prefix = atoms.size() - s.length();
		for (int j = 0; j < prefix; j++) {
			s = "0" + s;
		}
		return s;

	}

	public ArrayList<String> getAtoms() {
		return this.atoms;
	}

	public ArrayList<Double> getPi0() {
		return this.pi0;
	}

	public int getVariants() {
		return this.variant;
	}

	public int getTotalStates() {
		return this.totalStates;
	}

	public boolean isClassified() {
		return this.classified;
	}

	public ArrayList<Integer> getClassifiedAsPositive() {
		return this.classifiedAsPositive;
	}

	public ArrayList<ArrayList<String>> generateE() {
		return Utility.generatePowerSet(atoms);
	}

	public ArrayList<Integer> getClassifiedAtoms() {
		return this.classifiedAtoms;
	}

	public double[] getPosteriorProbabilityMap() {
		return this.posteriorProbabilityMap;
	}

	public int getTestCount() {
		return this.testCounter;
	}

	public ArrayList<Integer> getUpSet(int i) {
		ArrayList<Integer> ret = new ArrayList<>();
		ret.add(i);
		String str = intToBitStates(i);
		for (int j = i + 1; j < totalStates; j++) {
			boolean isUpSet = true;
			String s = intToBitStates(j);
			for (int k = 0; k < s.length(); k++) {
				if (s.charAt(k) < str.charAt(k)) {
					isUpSet = false;
					break;
				}
			}
			if (isUpSet)
				ret.add(j);
		}
		return ret;
	}

	public double[] generatePriorProbabilityMap() {
		double[] ret = new double[totalStates];
		for (int i = 0; i < totalStates; i++) {
			ret[i] = generatePriorProbability(i);
		}
		return ret;
	}

	public double generatePriorProbability(int i) {
		double prob = 1.0;
		String state = intToBitStates(i);
		for (int j = 0; j < state.length(); j++) {
			if (state.charAt(j) == '0') {
				prob *= pi0.get(j);
			} else if (state.charAt(j) == '1') {
				prob *= (1.0 - pi0.get(j));
			}
		}
		return prob;
	}

	public void resetTestCounter() {
		this.testCounter = 0;
	}

	public abstract void updatePosteriorProbability(String experimentInBit, String responsesInBit,
			double upsetThresholdUp, double upsetThresholdLo);

	public ArrayList<Integer> atomIndices() {
		ArrayList<Integer> ret = new ArrayList<>();
		for (int i = 0; i < pi0.size(); i++) {
			String s = "";
			for (int j = 0; j < pi0.size(); j++) {
				if (j != i)
					s += "0";
				if (j == i)
					s += "1";
			}
			ret.add(bitToInt(s));
		}
		return ret;
	}

	public void updateClassifiedAtomsAndClassifiedState(double upsetThresholdUp, double upsetThresholdLo) {
		ArrayList<Integer> atomsIndices = atomIndices();

		for (int atom : atomsIndices) {
			double probMass = getUpSetProbabilityMass(atom);
			boolean atomClassified;
			if (probMass <= 1 - upsetThresholdUp && probMass >= upsetThresholdLo) {
				atomClassified = false;
			} else {
				if (probMass < upsetThresholdLo) {
					if (!classifiedAsPositive.contains(atom))
						this.classifiedAsPositive.add(atom);
				}
				atomClassified = true;
			}
			if (!classifiedAtoms.contains(atom) && atomClassified)
				classifiedAtoms.add(atom);
			else if (classifiedAtoms.contains(atom) && !atomClassified) {
				for (int i = 0; i < classifiedAtoms.size(); i++) {
					if (classifiedAtoms.get(i) == atom) {
						classifiedAtoms.remove(i);
						break;
					}
				}
				for (int i = 0; i < classifiedAsPositive.size(); i++) {
					if (classifiedAsPositive.get(i) == atom) {
						classifiedAsPositive.remove(i);
						break;
					}
				}
			}
		}
		if (this.classifiedAtoms.size() == atoms.size() * variant) {
			this.classified = true;
		} else {
			this.classified = false;
		}
	}

	public double getUpSetProbabilityMass(int state) {
		double upProbability = 0.0;
		for (int i : getUpSet(state))
			upProbability += posteriorProbabilityMap[i];
		return upProbability;
	}

	public String findHalvingState(double prob) {
		String candidate = "abc";
		double min = Double.MAX_VALUE;
		for (int i = 1; i < (1 << atoms.size()); i++) {
			String[] partitionMap = new String[totalStates];
			for (int j = 0; j < totalStates; j++) {
				partitionMap[j] = ""; // otherwise initilized to null
			}

			String experimentInBit = intToBitExperiments(i);

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

			// for (int a = 0; a < totalStates; a++) {
			// System.out.print(bitToInt(partitionMap[a]) + " ");
			// }
			// System.out.println();

			double[] partitionProbMass = new double[(1 << variant)];
			for (int j = 0; j < totalStates; j++) {
				partitionProbMass[bitToInt(partitionMap[j])] += posteriorProbabilityMap[j];
			}
			double temp = 0.0;
			for (double d : partitionProbMass) {
				temp += Math.abs(d - prob);
			}
			if (temp < min) {
				min = temp;
				candidate = intToBitExperiments(i);
			}

		}

		return candidate;

	}

	public abstract double computeResponseProbabilityUsingTrueState(String experimentInBit, String responseInBit,
			String trueState);
}
