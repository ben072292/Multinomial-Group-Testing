package edu.cwru.bayesmultinomialtesting.tree.util;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.io.Serializable;

public class TreeStat implements Serializable {

    private double[] correctProb;
    private double[] incorrectProb;
    private double[] falsePositiveProb;
    private double[] falseNegativeProb;
    private double[] individualFalsePositive;
    private double[] individualFalseNegative;
    private double unclassifiedLeavesTotalProbability;
    private double expectedStages;
    private double expectedTests;
    private double stagesSD;
    private double testsSD;
    private int totalLeaves;
    private double[] runtimeRecord;

    public double[] getCorrectProb() {
        return this.correctProb;
    }

    public void setCorrectProb(double[] correctProb) {
        this.correctProb = correctProb;
    }

    public double[] getIncorrectProb() {
        return this.incorrectProb;
    }

    public void setIncorrectProb(double[] incorrectProb) {
        this.incorrectProb = incorrectProb;
    }

    public double[] getFalsePositiveProb() {
        return this.falsePositiveProb;
    }

    public void setFalsePositiveProb(double[] falsePositiveProb) {
        this.falsePositiveProb = falsePositiveProb;
    }

    public double[] getFalseNegativeProb() {
        return this.falseNegativeProb;
    }

    public void setFalseNegativeProb(double[] falseNegativeProb) {
        this.falseNegativeProb = falseNegativeProb;
    }

    public double[] getIndividualFalsePositive() {
        return this.individualFalsePositive;
    }

    public void setIndividualFalsePositive(double[] individualFalsePositive) {
        this.individualFalsePositive = individualFalsePositive;
    }

    public double[] getIndividualFalseNegative() {
        return this.individualFalseNegative;
    }

    public void setIndividualFalseNegative(double[] individualFalseNegative) {
        this.individualFalseNegative = individualFalseNegative;
    }

    public double getUnclassifiedLeavesTotalProbability() {
        return this.unclassifiedLeavesTotalProbability;
    }

    public void setUnclassifiedLeavesTotalProbability(double unclassifiedLeavesTotalProbability) {
        this.unclassifiedLeavesTotalProbability = unclassifiedLeavesTotalProbability;
    }

    public double getExpectedStages() {
        return this.expectedStages;
    }

    public void setExpectedStages(double expectedStages) {
        this.expectedStages = expectedStages;
    }

    public double getExpectedTests() {
        return this.expectedTests;
    }

    public void setExpectedTests(double expectedTests) {
        this.expectedTests = expectedTests;
    }

    public double getStagesSD() {
        return this.stagesSD;
    }

    public void setStagesSD(double stagesSD) {
        this.stagesSD = stagesSD;
    }

    public double getTestsSD() {
        return this.testsSD;
    }

    public void setTestsSD(double testsSD) {
        this.testsSD = testsSD;
    }

    public int getTotalLeaves() {
        return this.totalLeaves;
    }

    public void setTotalLeaves(int totalLeaves) {
        this.totalLeaves = totalLeaves;
    }

    public double[] getRuntimeRecord() {
        return runtimeRecord;
    }

    public void setRunTimeRecord(double[] runtimeRecord) {
        this.runtimeRecord = runtimeRecord;
    }

    private static void merge(double[] a, double[] b) {
        for (int i = 0; i < a.length; i++)
            a[i] += b[i];
    }

    public void mergeResults(TreeStat stat) {
        merge(correctProb, stat.getCorrectProb());
        merge(incorrectProb, stat.getIncorrectProb());
        merge(falsePositiveProb, stat.getFalsePositiveProb());
        merge(falseNegativeProb, stat.getFalseNegativeProb());
        merge(individualFalsePositive, stat.getIndividualFalsePositive());
        merge(individualFalseNegative, stat.getIndividualFalseNegative());
        this.expectedStages += stat.getExpectedStages();
        this.expectedTests += stat.getExpectedTests();
        this.stagesSD += stat.getStagesSD();
        this.testsSD += stat.getTestsSD();
        this.unclassifiedLeavesTotalProbability += stat.getUnclassifiedLeavesTotalProbability();
    }

    public void mergeRunTime(TreeStat stat) {
        merge(runtimeRecord, stat.getRuntimeRecord());
    }

    public void outputDetailedStat(int searchDepth, int k, int poolSize) {
        System.out.println("\n\nStatistics: \n\n");
        System.out.println("Average Leaves:," + getTotalLeaves() / Math.pow(2, poolSize));
        System.out.println("Stagewise Statistics");
        System.out.println("Stage,Classification,FP,FN");
        double correctSum = 0, wrongSum = 0, fpTotal = 0, fnTotal = 0;
        for (int i = 0; i < searchDepth; i++) {
            double tempCorrectProbTotal = 0;
            double tempWrongProbTotal = 0;
            double tempFPTotal = 0;
            double tempFNTotal = 0;
            for (int j = 1; j <= k; j++) {
                tempCorrectProbTotal += getCorrectProb()[i * k + j];
                tempWrongProbTotal += getIncorrectProb()[i * k + j];
                tempFPTotal += getFalsePositiveProb()[i * k + j];
                tempFNTotal += getFalseNegativeProb()[i * k + j];
            }
            correctSum += tempCorrectProbTotal;
            wrongSum += tempWrongProbTotal;
            fpTotal += tempFPTotal;
            fnTotal += tempFNTotal;

            System.out.println((i + 1) + "," + (tempCorrectProbTotal + tempWrongProbTotal) * 100 + " %,"
                    + tempFPTotal * 100 + " %," + tempFNTotal * 100 + " %");

        }
        System.out.println("Total," + (1 - getUnclassifiedLeavesTotalProbability()) * 100 + " %," + fpTotal * 100
                + " %," + fnTotal * 100 + " %\n\n");

        System.out.println("Potentially Can Be Classified in Future Stages:,"
                + getUnclassifiedLeavesTotalProbability() * 100 + " %");
        System.out.println("Total Probability Of Classified Sequences But With < .1 % Branch Probability:,"
                + (1 - getUnclassifiedLeavesTotalProbability() - (correctSum + wrongSum)) * 100 + " %");
        System.out.println("Expected Average Number Of Classification Stages:," + getExpectedStages());
        System.out.println("Expected Average Number Of Classification Tests:," + getExpectedTests());
        System.out.println("Standard Deviation For Number of Stages:," + getStagesSD());
        System.out.println("Standard Deviation For Number of Tests:," + getTestsSD());
    }

    public void outputSummaryStat(String identifier, int searchDepth, int k, int poolSize) {

        PrintStream out;
        try {
            out = new PrintStream(new FileOutputStream("statistics.csv", true));
            System.setOut(out);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        double correctSum = 0, wrongSum = 0;
        for (int i = 0; i < searchDepth; i++) {
            double tempCorrectProbTotal = 0;
            double tempWrongProbTotal = 0;
            for (int j = 1; j <= k; j++) {
                tempCorrectProbTotal += getCorrectProb()[i * k + j];
                tempWrongProbTotal += getIncorrectProb()[i * k + j];
            }
            correctSum += tempCorrectProbTotal;
            wrongSum += tempWrongProbTotal;

        }

        double fpTotal = 0, fnTotal = 0;
        for (int i = 0; i < searchDepth; i++) {
            double tempFPTotal = 0;
            double tempFNTotal = 0;
            for (int j = 1; j <= k; j++) {
                tempFPTotal += getFalsePositiveProb()[i * k + j];
                tempFNTotal += getFalseNegativeProb()[i * k + j];
            }
            fpTotal += tempFPTotal;
            fnTotal += tempFNTotal;

        }

        // System.out.println("Classification Rate, False Positive, False Negative,
        // Expected Stages, Expected Tests, Stages SD, Tests SD");
        System.out.println(
                identifier + "," + (correctSum + wrongSum) * 100 + " %," + fpTotal * 100 + " %," + fnTotal * 100 + " %,"
                        + getExpectedStages() + "," + getExpectedTests() + "," + getStagesSD() + "," + getTestsSD());
    }

    public void outputStat(String identifier, int searchDepth, int k, int poolSize, boolean outputSummary) {
        outputDetailedStat(searchDepth, k, poolSize);
        if (outputSummary)
            outputSummaryStat(identifier, searchDepth, k, poolSize);
    }
}
