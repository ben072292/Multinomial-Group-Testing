package edu.cwru.bayesmultinomialtestingbitwise.tests;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;

import edu.cwru.bayesmultinomialtestingbitwise.tree.util.TreeStat;
import edu.cwru.bayesmultinomialtestingbitwise.lattices.ProductLatticeBitwiseNonDilution;
import edu.cwru.bayesmultinomialtestingbitwise.tree.SingleTree;

public class SingleTreeTest {
    public static void main(String[] args) {
        int atoms = Integer.parseInt(args[0]);
        int variants = Integer.parseInt(args[1]);
        double prior = Double.parseDouble(args[2]);
        double classificationThresholdUp = 0.005;
        double classificationThresholdLo = 0.005;
        int searchDepth = 6;

        PrintStream out;
        try {
            out = new PrintStream(
                    new FileOutputStream("single_tree_analysis_bitwise_N=" + atoms + "_k=" + variants + "_prior=" + prior + "_depth=" + searchDepth + ".csv"));
            System.setOut(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        double[] pi0 = new double[atoms * variants];
        for (int i = 0; i < atoms * variants; i++)
            pi0[i] = prior;

        ProductLatticeBitwiseNonDilution p = new ProductLatticeBitwiseNonDilution(atoms, variants, pi0);

        // DFS
        SingleTree tree = new SingleTree(p, -1, -1, 1, 0, classificationThresholdUp, classificationThresholdLo, searchDepth);

        // BFS
        // SingleTree tree = new SingleTree(p, -1, -1, 0, true); // BFS
        // for(int i = 0; i < searchDepth; i++){
        //     tree.increaseStage(1, 0.01, 0.01, searchDepth);
        // }
        // ArrayList<SingleTree> leaves = new ArrayList<>();
        // SingleTree.findAll(tree, leaves);
        // for (SingleTree leaf : leaves) {
        //     leaf.setLattice(null);
        // }
        
        SingleTree st = tree.applyTrueState(p, 0, 0.001);

        TreeStat ret = st.parse(0, new ProductLatticeBitwiseNonDilution(p, 0), 0.001,
                1, searchDepth, pi0, 1.0);

        for (int i = 1; i < p.totalStates(); i++) {
            st = tree.applyTrueState(p, i, 0.001);
            TreeStat temp = st.parse(i, p, 0.001, 1, searchDepth, pi0, 1.0);
            ret.mergeResults(temp);
        }

        System.out.println("N = " + atoms + ", k = " + variants);
        System.out.print("Prior: ");
        for (int i = 0; i < atoms * variants; i++) {
            System.out.print(pi0[i] + ", ");
        }
        System.out.println();
        ret.outputStat("statistics", searchDepth, 1, atoms, false);

    }
}
