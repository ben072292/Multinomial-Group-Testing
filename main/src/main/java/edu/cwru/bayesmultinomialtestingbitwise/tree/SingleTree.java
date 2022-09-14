package edu.cwru.bayesmultinomialtestingbitwise.tree;

import edu.cwru.bayesmultinomialtestingbitwise.lattices.ProductLatticeBitwise;
import edu.cwru.bayesmultinomialtestingbitwise.lattices.ProductLatticeBitwiseNonDilution;
import edu.cwru.bayesmultinomialtestingbitwise.tree.util.TreeStat;

import java.util.ArrayList;

public class SingleTree {
	protected ProductLatticeBitwise lattice;
	protected int ex = -1;
	protected int res = -1;
	protected int exCount;
	protected double branchProb = 1.0;
	protected ArrayList<SingleTree> children = null;
	protected int curStage;
	protected boolean isClassified = false;
	protected int[] classificationStat;
	protected int latticeSize;

	/**
	 * Default Constructor
	 */
	public SingleTree() {
	}

	/**
	 * Simple constructor for {@link #increaseStage(int, double, double)}, which is
	 * BFS
	 * 
	 * @param lattice
	 * @param ex
	 * @param res
	 */
	public SingleTree(ProductLatticeBitwise lattice, int ex, int res, int currStage, boolean preserveLattice){
		if(preserveLattice) this.lattice = lattice;
		this.latticeSize = lattice.getAtoms();
		this.isClassified = lattice.isClassified();
		this.classificationStat = lattice.getClassificationStat();
		this.ex = ex;
		this.res = res;
		this.exCount = lattice.getTestCount();
		this.curStage = currStage;
	}

	/**
	 * Single-tree constructor (DFS).
	 * 
	 * @param lattice
	 * @param experiments
	 * @param experimentsResults
	 * @param k
	 * @param currStage
	 * @param upsetThresholdUp
	 * @param upsetThresholdLo
	 * @param stage
	 */
	public SingleTree(ProductLatticeBitwise lattice, int experiments, int experimentsResults, int k, int currStage,
			double upsetThresholdUp, double upsetThresholdLo, int stage) {
		this(lattice, experiments, experimentsResults, currStage, false);
		if (!lattice.isClassified() && currStage < stage) {
			this.children = new ArrayList<SingleTree>();
			int ex = lattice.findHalvingStatesParallel(1.0 / (1 << lattice.getVariants()));
			for (int res = 0; res < (1 << lattice.getVariants()); res++) {
				ProductLatticeBitwise p = new ProductLatticeBitwiseNonDilution(lattice, 1);
				p.updatePosteriorProbabilities(ex, res, upsetThresholdUp, upsetThresholdLo);
				this.addChild(new SingleTree(p, ex, res, k, currStage + 1, upsetThresholdUp, upsetThresholdLo, stage));

			}
		}
	}

	public void increaseStage(int k, double upsetThresholdUp, double upsetThresholdLo, int stage) {
		ArrayList<SingleTree> trees = new ArrayList<>();
		findClassified(this, trees);
		for (SingleTree tree : trees) {
			tree.setLattice(null);
		}
		trees.clear();
		findUnclassified(this, trees);
		for (SingleTree tree : trees) {
			if (tree.getCurrStage() < stage) {
				tree.setChildren(new ArrayList<SingleTree>());
				int childExperiments = tree.getLattice()
						.findHalvingStatesParallel(1.0 / (1 << tree.getLattice().getVariants()));
				for (int i = 0; i < (1 << tree.getLattice().getVariants()); i++) {
					ProductLatticeBitwise p = new ProductLatticeBitwiseNonDilution(tree.getLattice(), 1);
					p.updatePosteriorProbabilities(childExperiments, i, upsetThresholdUp, upsetThresholdLo);
					if(tree.curStage < stage-1)
						tree.addChild(new SingleTree(p, childExperiments, i, tree.getCurrStage() + 1, true));
					else if(tree.curStage == stage-1)
						tree.addChild(new SingleTree(p, childExperiments, i, tree.getCurrStage() + 1, false));
				}
				tree.setLattice(null);
				System.gc();
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
		this.exCount = old.getExCount();
		this.isClassified = old.isClassified();
		this.classificationStat = old.getClassifiedAtoms().clone();
		this.latticeSize = old.getLatticeSize(); // dramatically reduce time
		this.ex = old.getEx();
		this.res = old.getRes();
		this.branchProb = old.getBranchProb();
		this.curStage = old.getCurrStage();

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

	public int getCurrStage() {
		return this.curStage;
	}

	public int getEx() {
		return this.ex;
	}

	public int getRes() {
		return this.res;
	}

	public int getExCount() {
		return this.exCount;
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
			int k, int stage, double[] pi0, double symmetryCoef) {
		TreeStat ret = new TreeStat();
		double coef = originalLattice.generatePriorProbability(trueState, pi0) * symmetryCoef;
		double[] correctProb = new double[k * stage + 1];
		double[] incorrectProb = new double[k * stage + 1];
		double[] falsePositiveProb = new double[k * stage + 1];
		double[] falseNegativeProb = new double[k * stage + 1];
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
				correctProb[leaf.getExCount()] += leaf.getBranchProb() * coef;
			} else if (leaf.isClassified() && !leaf.isCorrectClassification(trueState)) {
				// System.out.println(leaf.getSelfProb() * 100);
				incorrectProb[leaf.getExCount()] += leaf.getBranchProb() * coef;
				falsePositiveProb[leaf.getExCount()] += leaf.falsePositive(trueState) * coef
						* leaf.getBranchProb();
				falseNegativeProb[leaf.getExCount()] += leaf.falseNegative(trueState) * coef
						* leaf.getBranchProb();
			} else if (!leaf.isClassified()) {
				unclassifiedLeavesTotalProbability += leaves.get(i).getBranchProb() * coef;
			}
			testLength[i] = leaf.getExCount();
			stageLength[i] = Math.ceil((double) testLength[i] / (double) k);
			expectedStages += stageLength[i] * leaf.getBranchProb();
			expectedTests += testLength[i] * leaf.getBranchProb();
		}

		for (int i = 0; i < size; i++) {
			SingleTree leaf = leaves.get(i);
			stagesSD += Math.pow((Math.ceil((double) testLength[i] / (double) k) - expectedStages), 2)
					* leaf.getBranchProb();
			testsSD += Math.pow((leaf.getExCount() - expectedTests), 2) * leaf.getBranchProb();
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
								node.getChildren().get(i).getEx(),
								node.getChildren().get(i).getRes(),
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
