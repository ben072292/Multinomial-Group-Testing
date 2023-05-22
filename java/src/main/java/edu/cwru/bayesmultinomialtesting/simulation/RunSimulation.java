package edu.cwru.bayesmultinomialtesting.simulation;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import edu.cwru.bayesmultinomialtesting.tree.util.TreeStat;
import edu.cwru.bayesmultinomialtesting.utilities.Pair;
import edu.cwru.bayesmultinomialtesting.lattices.ProductLatticeNonDilution;
import edu.cwru.bayesmultinomialtesting.tree.SingleTree;

public class RunSimulation {
    public static String intToBitStates(int i) {
        String s = Integer.toBinaryString(i);
        int prefix = 4 - s.length();
        for (int j = 0; j < prefix; j++) {
            s = "0" + s;
        }
        return s;

    }

    public static void main(String[] args) {

        PrintStream out;
        try {
            out = new PrintStream(new FileOutputStream("output.csv"));
            System.setOut(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        ArrayList<String> atom = new ArrayList<>();
        atom.add("A");
        atom.add("B");
        atom.add("C");

        ArrayList<Double> pi0 = new ArrayList<>();
        pi0.add(0.02);
        pi0.add(0.02);
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

        SingleTree tree = new SingleTree(p, null, null, 1, 0, 0.01, 0.01,
                searchDepth);

        String trueStateInBit = p.intToBitStates(0);
        SingleTree st = tree.applyTrueState(p, trueStateInBit, 0.001);

        TreeStat ret = st.parse(0, new ProductLatticeNonDilution(p, 0), 0.001, 1,
                searchDepth, 1.0);

        for (int i = 1; i < p.getTotalStates(); i++) {
            trueStateInBit = p.intToBitStates(i);
            st = tree.applyTrueState(p, trueStateInBit, 0.001);
            TreeStat temp = st.parse(i, p, 0.001, 1, searchDepth, 1.0);
            ret.mergeResults(temp);
        }

        ret.outputStat("haha", searchDepth, 1, atom.size(), false);

        System.out.println("\n Sequences: (0 means positive, 1 means negative");

        ArrayList<SingleTree> leaves = new ArrayList<>();
        SingleTree.findClassified(tree, leaves);
        System.out.println("Classified Leaves: " + leaves.size());

        for (SingleTree leaf : leaves) {
            for (Pair<String, String> pair : leaf.getLattice().experimentHistory) {
                System.out.print(pair.getFirst() + ": " + pair.getSecond() + " -> ");
            }
            System.out.println("*Classified");
        }

    }
}
