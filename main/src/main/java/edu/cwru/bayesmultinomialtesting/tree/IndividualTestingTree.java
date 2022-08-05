package edu.cwru.bayesmultinomialtesting.tree;

import java.util.ArrayList;

import edu.cwru.bayesmultinomialtesting.tree.util.TreeStat;
import edu.cwru.bayesmultinomialtesting.lattices.ProductLatticeNonDilution;

public class IndividualTestingTree {
	protected ProductLatticeNonDilution lattice;
	protected ArrayList<String> experiments;
	protected ArrayList<String> experimentsResults;
	protected int experimentSize;
	protected double selfProb = 1.0;
	protected ArrayList<IndividualTestingTree> children = null;
	protected int currentDepth;
	protected boolean isClassified = false;
	protected ArrayList<Integer> classifiedAsPositive = new ArrayList<>();
	protected int latticeSize;

	/**
	 * Default Constructor
	 */
	public IndividualTestingTree() {
	}

	/**
	 * Simple constructor for {@link #increaseDepth(int, double, double)}, which is
	 * BFS
	 * 
	 * @param lattice
	 * @param experiments
	 * @param experimentsResults
	 */
	public IndividualTestingTree(ProductLatticeNonDilution lattice, ArrayList<String> experiments,
			ArrayList<String> experimentsResults, int currentDepth) {
		this.lattice = lattice;
		this.latticeSize = lattice.getAtoms().size();
		this.isClassified = lattice.isClassified();
		this.classifiedAsPositive = lattice.getClassifiedAsPositive();
		this.experiments = experiments;
		this.experimentsResults = experimentsResults;
		this.experimentSize = lattice.getTestCounter();
		this.currentDepth = currentDepth;
	}

	/**
	 * Single-tree constructor (DFS).
	 * 
	 * @param lattice
	 * @param experiments
	 * @param experimentsResults
	 * @param k
	 * @param step
	 * @param upsetThresholdUp
	 * @param upsetThresholdLo
	 * @param searchDepth
	 */
	public IndividualTestingTree(ProductLatticeNonDilution lattice, ArrayList<String> experiments,
			ArrayList<String> experimentsResults, int k,
			int step, double upsetThresholdUp, double upsetThresholdLo, int searchDepth) {
		this(lattice, experiments, experimentsResults, step);
		if (!lattice.isClassified() && step < searchDepth) {
			this.children = new ArrayList<IndividualTestingTree>();
			ArrayList<String> childExperiments = individualTestingExperimentSelection(lattice);
			ArrayList<ArrayList<String>> childExperimentsResults = individualTestingExperimentResults(lattice);
			for (int i = 0; i < childExperimentsResults.size(); i++) {
				ProductLatticeNonDilution p = new ProductLatticeNonDilution(lattice, 1);
				for (int j = 0; j < childExperiments.size(); j++) {

					p.updatePosteriorProbability(childExperiments.get(j), childExperimentsResults.get(i).get(j),
							upsetThresholdUp, upsetThresholdLo);

				}
				this.addChild(new IndividualTestingTree(p, childExperiments,
						childExperimentsResults.get(i), k, step + 1, upsetThresholdUp, upsetThresholdLo, searchDepth));

			}
		}
	}

	public ArrayList<String> individualTestingExperimentSelection(ProductLatticeNonDilution lattice) {
		ArrayList<String> ret = new ArrayList<>();
		for (int i = 0; i < lattice.getAtoms().size(); i++) {
			ret.add(lattice.intToBitExperiments((1 << i)));
		}
		return ret;
	}

	public ArrayList<ArrayList<String>> individualTestingExperimentResults(ProductLatticeNonDilution lattice) {
		ArrayList<ArrayList<String>> ret = new ArrayList<>();

		for (int i = 0; i < lattice.getTotalStates(); i++) {
			String s = lattice.intToBitStates(i);
			ArrayList<String> temp = new ArrayList<>();
			for (int j = 0; j < lattice.getAtoms().size(); j++) {
				String ss = "";
				for (int k = 0; k < lattice.getVariants(); k++) {
					ss += s.charAt(j * lattice.getVariants() + k);
				}
				temp.add(ss);
			}
			ret.add(temp);
		}

		return ret;
	}

	/**
	 * Copy constructor
	 * 
	 * @param old
	 * @param recursiveCopy
	 */
	public IndividualTestingTree(IndividualTestingTree old) {
		if (old.getLattice() != null)
			this.lattice = new ProductLatticeNonDilution(old.getLattice(), 2); // use light constructor
		this.experimentSize = old.getExperimentSize();
		this.isClassified = old.isClassified();
		this.classifiedAsPositive = new ArrayList<>(old.getClassifiedAsPositive());
		this.latticeSize = old.getLatticeSize(); // dramatically reduce time
		if (old.getExperiments() != null)
			this.experiments = old.getExperiments();
		if (old.getExperimentsResults() != null) {
			this.experimentsResults = old.getExperimentsResults();
		}
		this.selfProb = old.getSelfProb();
		this.currentDepth = old.getCurrentDepth();

	}

	/**
	 * Copy constructor
	 * 
	 * @param old
	 * @param recursiveCopy
	 */
	public IndividualTestingTree(IndividualTestingTree old, boolean recursiveCopy) {
		this(old);
		if (recursiveCopy) {
			if (old.getChildren() != null) {
				this.children = new ArrayList<>();
				for (IndividualTestingTree oldChild : old.getChildren()) {
					this.addChild(new IndividualTestingTree(oldChild, recursiveCopy));
				}
			}
		}
	}

	public double getSelfProb() {
		return this.selfProb;
	}

	public ProductLatticeNonDilution getLattice() {
		return this.lattice;
	}

	public int getCurrentDepth() {
		return this.currentDepth;
	}

	public ArrayList<String> getExperiments() {
		return this.experiments;
	}

	public ArrayList<String> getExperimentsResults() {
		return this.experimentsResults;
	}

	public int getExperimentSize() {
		return this.lattice.getTestCounter();
	}

	public ArrayList<IndividualTestingTree> getChildren() {
		return this.children;
	}

	public int getLatticeSize() {
		return this.latticeSize;
	}

	public boolean isClassified() {
		return this.isClassified;
	}

	public ArrayList<Integer> getClassifiedAsPositive() {
		return this.classifiedAsPositive;
	}

	public void addChild(IndividualTestingTree child) {
		this.children.add(child);
	}

	public void setChildren(ArrayList<IndividualTestingTree> children) {
		this.children = children;
	}

	public void setLattice(ProductLatticeNonDilution lattice) {
		this.lattice = lattice;
	}

	public void pruneLattice() {
		this.lattice = null;
	}

	public TreeStat parse(int trueState, ProductLatticeNonDilution originalLattice, double selfProbabilityThreshold,
			int k, int searchDepth, double symmetryCoef) {
		TreeStat ret = new TreeStat();
		double coef = originalLattice.generatePriorProbability(trueState) * symmetryCoef;
		double[] correctProb = new double[k * searchDepth + 1];
		double[] incorrectProb = new double[k * searchDepth + 1];
		double[] falsePositiveProb = new double[k * searchDepth + 1];
		double[] falseNegativeProb = new double[k * searchDepth + 1];
		double[] individualFalsePositive = new double[originalLattice.getAtoms().size()];
		double[] individualFalseNegative = new double[originalLattice.getAtoms().size()];
		double unclassifiedLeavesTotalProbability = 0.0;
		double expectedStages = 0.0;
		double expectedTests = 0.0;
		double stagesSD = 0.0;
		double testsSD = 0.0;
		int totalLeaves = 0;

		ArrayList<IndividualTestingTree> leaves = new ArrayList<>();
		findAll(this, leaves);
		int size = leaves.size();
		totalLeaves = size;
		double[] stageLength = new double[size];
		double[] testLength = new double[size];
		for (int i = 0; i < size; i++) {
			IndividualTestingTree leaf = leaves.get(i);
			if (leaf.isClassified() && leaf.isCorrectClassification(trueState)) {
				correctProb[leaf.getExperimentSize()] += leaf.getSelfProb() * coef;
			} else if (leaf.isClassified() && !leaf.isCorrectClassification(trueState)) {
				// System.out.println(leaf.getSelfProb() * 100);
				incorrectProb[leaf.getExperimentSize()] += leaf.getSelfProb() * coef;
				falsePositiveProb[leaf.getExperimentSize()] += leaf.falsePositive(trueState) * coef
						* leaf.getSelfProb();
				falseNegativeProb[leaf.getExperimentSize()] += leaf.falseNegative(trueState) * coef
						* leaf.getSelfProb();
			} else if (!leaf.isClassified()) {
				unclassifiedLeavesTotalProbability += leaves.get(i).getSelfProb() * coef;
			}
			testLength[i] = leaf.getExperimentSize();
			stageLength[i] = Math.ceil((double) testLength[i] / (double) k);
			expectedStages += stageLength[i] * leaf.getSelfProb();
			expectedTests += testLength[i] * leaf.getSelfProb();
		}

		for (int i = 0; i < size; i++) {
			IndividualTestingTree leaf = leaves.get(i);
			stagesSD += Math.pow((Math.ceil((double) testLength[i] / (double) k) - expectedStages), 2)
					* leaf.getSelfProb();
			testsSD += Math.pow((leaf.getExperimentSize() - expectedTests), 2) * leaf.getSelfProb();
		}
		stagesSD = Math.sqrt(stagesSD) * coef;
		testsSD = Math.sqrt(testsSD) * coef;
		expectedStages *= coef;
		expectedTests *= coef;

		ret.setCorrectProb(correctProb);
		ret.setIncorrectProb(incorrectProb);
		ret.setFalsePositiveProb(falsePositiveProb);
		ret.setFalseNegativeProb(falseNegativeProb);
		ret.setIndividualFalsePositive(individualFalsePositive);
		ret.setIndividualFalseNegative(individualFalseNegative);
		ret.setUnclassifiedLeavesTotalProbability(unclassifiedLeavesTotalProbability);
		ret.setExpectedStages(expectedStages);
		ret.setExpectedTests(expectedTests);
		ret.setStagesSD(stagesSD);
		ret.setTestsSD(testsSD);
		ret.setTotalLeaves(totalLeaves);

		return ret;
	}

	public boolean isCorrectClassification(int trueState) {
		String trueStateInBit = this.getLattice().intToBitStates(trueState);
		int temp = 0;
		for (int positive : this.getClassifiedAsPositive()) {
			temp += positive;
		}
		String atomsActivationInBit = this.getLattice().intToBitStates(this.getLattice().getTotalStates() - 1 - temp);
		return atomsActivationInBit.equals(trueStateInBit) ? true : false;
	}

	public int falsePositive(int trueState) {
		String trueStateInBit = this.getLattice().intToBitStates(trueState);
		int temp = 0;
		for (int positive : this.getClassifiedAsPositive()) {
			temp += positive;
		}
		String atomsActivationInBit = this.getLattice().intToBitStates(this.getLattice().getTotalStates() - 1 - temp);
		for (int i = 0; i < atomsActivationInBit.length(); i++) {
			if (atomsActivationInBit.charAt(i) == '0' && trueStateInBit.charAt(i) == '1')
				return 1;
		}
		return 0;

	}

	public int falseNegative(int trueState) {
		String trueStateInBit = this.getLattice().intToBitStates(trueState);
		int temp = 0;
		for (int positive : this.getClassifiedAsPositive()) {
			temp += positive;
		}
		String atomsActivationInBit = this.getLattice().intToBitStates(this.getLattice().getTotalStates() - 1 - temp);
		for (int i = 0; i < atomsActivationInBit.length(); i++) {
			if (atomsActivationInBit.charAt(i) == '1' && trueStateInBit.charAt(i) == '0')
				return 1;
		}
		return 0;

	}

	public void setSelfProb(double prob) {
		this.selfProb = prob;
	}

	/**
	 * Recursively find all leaves in the simulation tree.
	 * 
	 * @param node
	 * @param leaves
	 */
	public static void findAll(IndividualTestingTree node, ArrayList<IndividualTestingTree> leaves) {
		if (node == null) {
			return;
		} else {
			if (node.getChildren() == null) {
				leaves.add(node);
			} else {
				for (IndividualTestingTree child : node.getChildren()) {
					findAll(child, leaves);
				}
			}
		}
	}

	/**
	 * Recursively find all classified leaves in the simulation tree.
	 * 
	 * @param node
	 * @param leaves
	 */
	public static void findClassified(IndividualTestingTree node, ArrayList<IndividualTestingTree> leaves) {
		if (node == null) {
			return;
		}
		if (node.isClassified()) {
			leaves.add(node);
		} else {
			if (node.getChildren() != null) {
				for (IndividualTestingTree child : node.getChildren()) {
					findClassified(child, leaves);
				}
			}
		}
	}

	/**
	 * Find all the yet to classified leaves in the simulation tree.
	 * 
	 * @param node
	 * @param leaves
	 */
	public static void findUnclassified(IndividualTestingTree node, ArrayList<IndividualTestingTree> leaves) {
		if (node == null || node.isClassified()) {
			return;
		}
		if (node.getChildren() == null && !node.isClassified()) {
			leaves.add(node);
		} else if (node.getChildren() != null && !node.isClassified()) {
			for (IndividualTestingTree ks : node.getChildren()) {
				findUnclassified(ks, leaves);
			}
		}
	}

	public IndividualTestingTree applyTrueState(ProductLatticeNonDilution originalLattice, String trueState,
			double selfProbabilityThreshold) {
		IndividualTestingTree newTree = new IndividualTestingTree(this, false);
		applyTrueStateHelper(originalLattice, newTree, this, trueState, 1.0, selfProbabilityThreshold);
		return newTree;
	}

	/**
	 * Helper function for {@link #applyTrueState(ArrayList, double)}
	 * 
	 * @param ret
	 * @param node
	 * @param newTrueState
	 * @param branchProbability
	 * @param branchProbabilityThreshold
	 */
	public static void applyTrueStateHelper(ProductLatticeNonDilution originalLattice, IndividualTestingTree ret,
			IndividualTestingTree node, String trueState, double branchProbability,
			double branchProbabilityThreshold) {
		if (node == null)
			return;
		// if(node.getChildren() == null){
		// ret.setLattice(node.getLatticeFactory().copy(node.getLattice(), 2)); // use
		// light copy
		// }
		ret.setSelfProb(branchProbability);
		if (node.getChildren() != null) {

			for (int i = 0; i < node.getChildren().size(); i++) {
				double childProbability = ret.getSelfProb();
				if (ret.getChildren() == null)
					ret.setChildren(new ArrayList<>());
				for (int j = 0; j < node.getChildren().get(i).getExperiments().size(); j++) {
					childProbability *= originalLattice.computeResponseProbabilityUsingTrueState(
							node.getChildren().get(i).getExperiments().get(j),
							node.getChildren().get(i).getExperimentsResults().get(j), trueState);
				}

				ret.getChildren().add(new IndividualTestingTree(node.getChildren().get(i), false));
				if (childProbability > branchProbabilityThreshold) {
					applyTrueStateHelper(originalLattice, ret.getChildren().get(i), node.getChildren().get(i),
							trueState, childProbability,
							branchProbabilityThreshold);
				} else {
					ret.getChildren().set(i, null);
				}
			}
		}
	}
}
