package edu.cwru.bayesmultinomialtesting.utilities;

import java.util.ArrayList;
import java.util.stream.IntStream;

/**
 * Utility functions for constructing Poset models
 * 
 * @author Ben
 *
 */
public class Utility {

	/**
	 * Generate {@link #dilutionMatrix} using standard formula
	 * 
	 * @param alpha
	 * @param h
	 */
	public static ArrayList<ArrayList<Double>> generateDilutionMatrix(int n, double alpha, double h) {
		ArrayList<ArrayList<Double>> ret = new ArrayList<>();
		for (int rk = 1; rk <= n; rk++) {
			ArrayList<Double> table = new ArrayList<>();
			table.add(alpha);
			for (int r = 1; r <= rk; r++) {
				int k = rk - r;
				table.add(1 - alpha * r / (k * h + r));
			}
			ret.add(table);
		}
		return ret;

	}

	/**
	 * Generate power set index of selected size n
	 * 
	 * @param n
	 * @return List of power set index of selected size n
	 */
	public static ArrayList<ArrayList<Integer>> generatePowerSetIndex(int n) {
		ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
		int pow_set_size = (1 << n);
		int i, j;
		for (i = 0; i < pow_set_size; i++) {
			ArrayList<Integer> temp = new ArrayList<>();
			for (j = 0; j < n; j++) {
				/*
				 * Check if j-th bit in the counter is set If set then print j-th element from
				 * set
				 */
				if ((i & (1 << j)) > 0) {
					temp.add(1);
				} else {
					temp.add(0);
				}
			}
			ret.add(temp);
		}
		// System.out.println(ret);
		return ret;
	}

	/**
	 * Generate the power set lattice.
	 * 
	 * @param <T>
	 * @param N
	 * @return List of power set lattice.
	 */
	public static <T> ArrayList<ArrayList<T>> generatePowerSet(ArrayList<T> N) {
		int set_size = N.size();
		/* set_size of power set of a set with set_size n is (2**n -1) */
		int pow_set_size = (1 << set_size);
		int i, j;
		ArrayList<ArrayList<T>> ret = new ArrayList<>(pow_set_size);
		/* Run from counter 000..0 to 111..1 */
		for (i = 0; i < pow_set_size; i++) {
			ArrayList<T> temp = new ArrayList<>();
			for (j = 0; j < set_size; j++) {
				/*
				 * Check if j-th bit in the counter is set If set then print j-th element from
				 * set
				 */
				if ((i & (1 << j)) > 0) {
					temp.add(N.get(j));
				}
			}
			ret.add(temp);
		}
		return ret;
	}

	/**
	 * Generate the power set lattice in parallel.
	 * 
	 * @param <T>
	 * @param N
	 * @return List of power set lattice.
	 */
	public static <T> ArrayList<ArrayList<T>> generatePowerSetConcurrent(ArrayList<T> N) {
		int set_size = N.size();
		/* set_size of power set of a set with set_size n is (2**n -1) */
		int pow_set_size = (1 << set_size);
		ArrayList<ArrayList<T>> ret = new ArrayList<>();
		for (int i = 0; i < pow_set_size; i++) {
			ret.add(new ArrayList<T>());
		}
		/* Run from counter 000..0 to 111..1 */
		IntStream.range(1, pow_set_size).parallel().forEach(i -> {
			ArrayList<T> temp = new ArrayList<>();
			for (int j = 0; j < set_size; j++) {
				/*
				 * Check if j-th bit in the counter is set If set then print j-th element from
				 * set
				 */
				if ((i & (1 << j)) > 0) {
					temp.add(N.get(j));
				}
			}
			ret.set(i, temp);
		});
		return ret;
	}

	public static void individualFalsePositive(double[] falsePositiveCounterList, ArrayList<String> total,
			ArrayList<String> trueState, ArrayList<String> classifiedAsPositive, double coef) {
		ArrayList<String> falsePositive = new ArrayList<>(classifiedAsPositive);
		for (String ts : trueState) {
			if (classifiedAsPositive.contains(ts)) {
				falsePositive.remove(ts);
			}
		}
		for (int i = 0; i < total.size(); i++) {
			if (falsePositive.contains(total.get(i))) {
				// falsePositiveCounterList[i] += coef;
				falsePositiveCounterList[i] = coef;
			}
		}
	}

	public static void individualfalseNegative(double[] falseNegativeCounterList, ArrayList<String> total,
			ArrayList<String> trueState, ArrayList<String> classifiedAsPositive, double coef) {
		ArrayList<String> falseNegative = new ArrayList<>();
		ArrayList<String> classifiedAsNegative = new ArrayList<>(total);
		classifiedAsNegative.removeAll(classifiedAsPositive);
		for (String ts : trueState) {
			if (classifiedAsNegative.contains(ts)) {
				falseNegative.add(ts);
			}
		}
		for (int i = 0; i < total.size(); i++) {
			if (falseNegative.contains(total.get(i))) {
				falseNegativeCounterList[i] += coef;
				falseNegativeCounterList[i] = coef;
			}
		}
	}

	public static long binomial(int n, int k) {
		if (k == 0)
			return 1;
		return (n * binomial(n - 1, k - 1)) / k;
	}

	public static void main(String[] args) {

	}
}
