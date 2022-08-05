package edu.cwru.bayesmultinomialtesting.simulation;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import edu.cwru.bayesmultinomialtesting.tree.util.TreeStat;
import edu.cwru.bayesmultinomialtesting.lattices.ProductLatticeNonDilution;
import edu.cwru.bayesmultinomialtesting.tree.IndividualTestingTree;

public class RunIndividualTestingSimulation {
    public static void main(String[] args) {

        PrintStream out;
        try {
            out = new PrintStream(new FileOutputStream("individual-test.csv"));
            System.setOut(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        ArrayList<String> atom = new ArrayList<>();
        atom.add("A");
        atom.add("B");

        ArrayList<Double> pi0 = new ArrayList<>();
        pi0.add(0.02);
        pi0.add(0.02);
        pi0.add(0.02);
        pi0.add(0.02);

        ProductLatticeNonDilution p = new ProductLatticeNonDilution(atom, pi0);

        // System.out.println(p.getUpSetProbabilityMass(8) + " " +
        // p.getUpSetProbabilityMass(4) + " " + p.getUpSetProbabilityMass(2) + " " +
        // p.getUpSetProbabilityMass(1) + " " );

        // p.updatePosteriorProbability("11", "00", 0.01, 0.01);
        // System.out.println(p.getUpSetProbabilityMass(8) + " " +
        // p.getUpSetProbabilityMass(4) + " " + p.getUpSetProbabilityMass(2) + " " +
        // p.getUpSetProbabilityMass(1) + " " );
        // System.out.println(p.isClassified());

        // p.updatePosteriorProbability("11", "01", 0.01, 0.01);
        // System.out.println(p.getUpSetProbabilityMass(8) + " " +
        // p.getUpSetProbabilityMass(4) + " " + p.getUpSetProbabilityMass(2) + " " +
        // p.getUpSetProbabilityMass(1) + " " );
        // System.out.println(p.isClassified());

        // p.updatePosteriorProbability("01", "11", 0.01, 0.01);
        // System.out.println(p.getUpSetProbabilityMass(8) + " " +
        // p.getUpSetProbabilityMass(4) + " " + p.getUpSetProbabilityMass(2) + " " +
        // p.getUpSetProbabilityMass(1) + " " );
        // System.out.println(p.isClassified());

        // p.updatePosteriorProbability("10", "10", 0.01, 0.01);
        // System.out.println(p.getUpSetProbabilityMass(8) + " " +
        // p.getUpSetProbabilityMass(4) + " " + p.getUpSetProbabilityMass(2) + " " +
        // p.getUpSetProbabilityMass(1) + " " );
        // System.out.println(p.isClassified());

        // p.updatePosteriorProbability("11", "10", 0.01, 0.01);
        // System.out.println(p.getUpSetProbabilityMass(8) + " " +
        // p.getUpSetProbabilityMass(4) + " " + p.getUpSetProbabilityMass(2) + " " +
        // p.getUpSetProbabilityMass(1) + " " );
        // System.out.println(p.isClassified());

        // for(Pair<String, String> pair : p.experimentHistory){
        // System.out.print(pair.getFirst() + ": " + pair.getSecond() + " -> ");
        // }
        // System.out.println();

        int searchDepth = 5;
        int k = 2;
        double branchProbability = 0.001;

        IndividualTestingTree tree = new IndividualTestingTree(p, null, null, k, 0, 0.01, 0.01, searchDepth);

        String trueStateInBit = p.intToBitStates(0);
        IndividualTestingTree st = tree.applyTrueState(p, trueStateInBit, branchProbability);
        TreeStat ret = st.parse(0, new ProductLatticeNonDilution(p, 0), branchProbability, k, searchDepth, 1.0);

        for (int i = 1; i < 16; i++) {
            trueStateInBit = p.intToBitStates(i);
            st = tree.applyTrueState(p, trueStateInBit, branchProbability);
            TreeStat temp = st.parse(i, p, branchProbability, k, searchDepth, 1.0);
            ret.mergeResults(temp);
        }

        ret.outputStat("haha", searchDepth, k, atom.size(), false);

        // System.out.println("\n Sequences: (0 means positive, 1 means negative");

        // ArrayList<IndividualTestingTree> leaves = new ArrayList<>();
        // IndividualTestingTree.findClassified(tree, leaves);

        // for(IndividualTestingTree leaf : leaves){
        // for(Pair<String, String> pair : leaf.getLattice().experimentHistory){
        // System.out.print(pair.getFirst() + ": " + pair.getSecond() + " -> ");
        // }
        // System.out.println("*Classified");
        // }

    }
}
