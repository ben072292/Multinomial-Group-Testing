package edu.cwru.bayesmultinomialtestingbitwise.tree;

import edu.cwru.bayesmultinomialtestingbitwise.lattices.ProductLatticeBitwise;
import edu.cwru.bayesmultinomialtestingbitwise.lattices.ProductLatticeBitwiseNonDilution;
import edu.cwru.bayesmultinomialtestingbitwise.tree.util.TreeStat;

import java.util.ArrayList;

public class SingleTree {
	protected ProductLatticeBitwise lattice;
	protected int experiments = -1;
	protected int experimentsResponses = -1;
	protected int testCount;
	protected double branchProb = 1.0;
	protected ArrayList<SingleTree> children = null;
	protected int currentDepth;
	protected boolean isClassified = false;
	protected int[] classificationStat;
	protected int latticeSize;

	/**
	 * Default Constructor
	 */
	public SingleTree() {
	}

	/**
	 * Simple constructor for {@link #increaseDepth(int, double, double)}, which is
	 * BFS
	 * 
	 * @param lattice
	 * @param experiments
	 * @param experimentsResults
	 */
	public SingleTree(ProductLatticeBitwise lattice, int experiments, int experimentsResults,
			int currentDepth) {
		this.lattice = lattice;
		this.latticeSize = lattice.getAtoms();
		this.isClassified = lattice.isClassified();
		this.classificationStat = lattice.getClassificationStat();
		this.experiments = experiments;
		this.experimentsResponses = experimentsResults;
		this.testCount = lattice.getTestCount();
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
	public SingleTree(ProductLatticeBitwise lattice, int experiments, int experimentsResults, int k,
			int step, double upsetThresholdUp, double upsetThresholdLo, int searchDepth) {
		this(lattice, experiments, experimentsResults, step);
		if (!lattice.isClassified() && step < searchDepth) {
			this.children = new ArrayList<SingleTree>();
			int childExperiments = this.lattice.findHalvingStatesParallel(1.0 / (1 << lattice.getVariants()));
			for (int i = 0; i < (1 << lattice.getVariants()); i++) {
				ProductLatticeBitwise p = new ProductLatticeBitwiseNonDilution(lattice, 1);
				p.updatePosteriorProbabilities(childExperiments, i, upsetThresholdUp, upsetThresholdLo);
				this.addChild(new SingleTree(p, childExperiments, i, k, step + 1, upsetThresholdUp, upsetThresholdLo,
						searchDepth));

			}
		}
	}

	/**
	 * Copy constructor
	 * 
	 * @param old
	 * @param recursiveCopy
	 */
	public SingleTree(SingleTree old) {
		if (old.getLattice() != null)
			this.lattice = new ProductLatticeBitwiseNonDilution(old.getLattice(), 2); // use light constructor
		this.testCount = old.getExperimentSize();
		this.isClassified = old.isClassified();
		this.classificationStat = old.getClassifiedAtoms().clone();
		this.latticeSize = old.getLatticeSize(); // dramatically reduce time
		this.experiments = old.getExperiments();
		this.experimentsResponses = old.getExperimentResponses();
		this.branchProb = old.getBranchProb();
		this.currentDepth = old.getCurrentDepth();

	}

	/**
	 * Copy constructor
	 * 
	 * @param old
	 * @param recursiveCopy
	 */
	public SingleTree(SingleTree old, boolean recursiveCopy) {
		this(old);
		if (recursiveCopy) {
			if (old.getChildren() != null) {
				this.children = new ArrayList<>();
				for (SingleTree oldChild : old.getChildren()) {
					this.addChild(new SingleTree(oldChild, recursiveCopy));
				}
			}
		}
	}

	public double getBranchProb() {
		return this.branchProb;
	}

	public ProductLatticeBitwise getLattice() {
		return this.lattice;
	}

	public int getCurrentDepth() {
		return this.currentDepth;
	}

	public int getExperiments() {
		return this.experiments;
	}

	public int getExperimentResponses() {
		return this.experimentsResponses;
	}

	public int getExperimentSize() {
		return this.testCount;
	}

	public ArrayList<SingleTree> getChildren() {
		return this.children;
	}

	public int getLatticeSize() {
		return this.latticeSize;
	}

	public boolean isClassified() {
		return this.isClassified;
	}

	public int[] getClassifiedAtoms() {
		return this.classificationStat;
	}

	public void addChild(SingleTree child) {
		this.children.add(child);
	}

	public void setChildren(ArrayList<SingleTree> children) {
		this.children = children;
	}

	public void setLattice(ProductLatticeBitwise lattice) {
		this.lattice = lattice;
	}

	public void pruneLattice() {
		this.lattice = null;
	}

	public TreeStat parse(int trueState, ProductLatticeBitwise originalLattice,
			double selfProbabilityThreshold,
			int k, int searchDepth, double[] pi0, double symmetryCoef) {
		TreeStat ret = new TreeStat();
		double coef = originalLattice.generatePriorProbability(trueState, pi0) * symmetryCoef;
		double[] correctProb = new double[k * searchDepth + 1];
		double[] incorrectProb = new double[k * searchDepth + 1];
		double[] falsePositiveProb = new double[k * searchDepth + 1];
		double[] falseNegativeProb = new double[k * searchDepth + 1];
		double[] individualFalsePositive = new double[originalLattice.getAtoms()];
		double[] individualFalseNegative = new double[originalLattice.getAtoms()];
		double unclassifiedLeavesTotalProbability = 0.0;
		double expectedStages = 0.0;
		double expectedTests = 0.0;
		double stagesSD = 0.0;
		double testsSD = 0.0;
		int totalLeaves = 0;

		ArrayList<SingleTree> leaves = new ArrayList<>();
		findAll(this, leaves);
		int size = leaves.size();
		totalLeaves = size;
		double[] stageLength = new double[size];
		double[] testLength = new double[size];
		for (int i = 0; i < size; i++) {
			SingleTree leaf = leaves.get(i);
			if (leaf.isClassified() && leaf.isCorrectClassification(trueState)) {
				correctProb[leaf.getExperimentSize()] += leaf.getBranchProb() * coef;
			} else if (leaf.isClassified() && !leaf.isCorrectClassification(trueState)) {
				// System.out.println(leaf.getSelfProb() * 100);
				incorrectProb[leaf.getExperimentSize()] += leaf.getBranchProb() * coef;
				falsePositiveProb[leaf.getExperimentSize()] += leaf.falsePositive(trueState) * coef
						* leaf.getBranchProb();
				falseNegativeProb[leaf.getExperimentSize()] += leaf.falseNegative(trueState) * coef
						* leaf.getBranchProb();
			} else if (!leaf.isClassified()) {
				unclassifiedLeavesTotalProbability += leaves.get(i).getBranchProb() * coef;
			}
			testLength[i] = leaf.getExperimentSize();
			stageLength[i] = Math.ceil((double) testLength[i] / (double) k);
			expectedStages += stageLength[i] * leaf.getBranchProb();
			expectedTests += testLength[i] * leaf.getBranchProb();
		}

		for (int i = 0; i < size; i++) {
			SingleTree leaf = leaves.get(i);
			stagesSD += Math.pow((Math.ceil((double) testLength[i] / (double) k) - expectedStages), 2)
					* leaf.getBranchProb();
			testsSD += Math.pow((leaf.getExperimentSize() - expectedTests), 2) * leaf.getBranchProb();
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

	public int actualTrueState() {
		int actual = 0;
		for (int i = 0; i < classificationStat.length; i++) {
			if (classificationStat[i] == 1)
				actual += (1 << i);
		}
		return actual;
	}

	public double totalPositive() {
		return classificationStat.length - Long.bitCount(actualTrueState());
	}

	public double totalNegative() {
		return Long.bitCount(actualTrueState());
	}

	public boolean isCorrectClassification(int trueState) {
		return actualTrueState() == trueState;
	}

	public double falsePositive(int trueState) {
		return totalPositive() == 0.0 ? 0.0
				: Long.bitCount(actualTrueState() ^ trueState | trueState) / totalPositive();

	}

	public double falseNegative(int trueState) {
		return totalNegative() == 0.0 ? 0.0
				: Long.bitCount(actualTrueState() ^ trueState | actualTrueState()) / totalNegative();
	}

	public void setSelfProb(double prob) {
		this.branchProb = prob;
	}

	/**
	 * Recursively find all leaves in the simulation tree.
	 * 
	 * @param node
	 * @param leaves
	 */
	public static void findAll(SingleTree node, ArrayList<SingleTree> leaves) {
		if (node == null) {
			return;
		} else {
			if (node.getChildren() == null) {
				leaves.add(node);
			} else {
				for (SingleTree child : node.getChildren()) {
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
	public static void findClassified(SingleTree node, ArrayList<SingleTree> leaves) {
		if (node == null) {
			return;
		}
		if (node.isClassified()) {
			leaves.add(node);
		} else {
			if (node.getChildren() != null) {
				for (SingleTree child : node.getChildren()) {
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
	public static void findUnclassified(SingleTree node, ArrayList<SingleTree> leaves) {
		if (node == null || node.isClassified()) {
			return;
		}
		if (node.getChildren() == null && !node.isClassified()) {
			leaves.add(node);
		} else if (node.getChildren() != null && !node.isClassified()) {
			for (SingleTree ks : node.getChildren()) {
				findUnclassified(ks, leaves);
			}
		}
	}

	public SingleTree applyTrueState(ProductLatticeBitwise originalLattice, int trueState,
			double selfProbabilityThreshold) {
		SingleTree newTree = new SingleTree(this, false);
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
	public static void applyTrueStateHelper(ProductLatticeBitwise originalLattice, SingleTree ret,
			SingleTree node, int trueState, double branchProbability, double branchProbabilityThreshold) {
		if (node == null)
			return;
		ret.setSelfProb(branchProbability);
		if (node.getChildren() != null) {
			for (int i = 0; i < node.getChildren().size(); i++) {
				if (ret.getChildren() == null)
					ret.setChildren(new ArrayList<>());
				double childProbability = ret.getBranchProb()
						* originalLattice.responseProbability(
								node.getChildren().get(i).getExperiments(),
								node.getChildren().get(i).getExperimentResponses(),
								trueState);

				ret.getChildren().add(new SingleTree(node.getChildren().get(i), false));
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
