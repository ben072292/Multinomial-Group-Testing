package edu.cwru.bayesmultinomialtestingbitwise.lattices;

public interface ProductLatticeBitwise {
    int getAtoms();

    void setAtoms(int atoms);

    int getVariants();

    void setVariants(int variants);

    int[] getClassificationStat();

    void setClassificationStat(int[] classificationStat);

    void copyClassificationStat(int[] classificationStat);

    double[] getPosteriorProbabilities();

    void setPosteriorProbabilities(double[] posteriorProbabilities);

    void copyPosteriorProbabilities(double[] posteriorProbabilities);

    int getTestCount();

    void setTestCount(int testCount);

    int totalStates();

    int nominalPoolSize();

    int[] getUpSet(int state);

    double[] generatePriorProbabilities(double[] pi0);

    double generatePriorProbability(int state, double[] pi0);

    void resetTestCount();

    void updatePosteriorProbabilities(int experiment, int response, double upsetThresholdUp, double upsetThresholdLo);

    void updatePosteriorProbabilitiesParallel(int experiment, int response, double upsetThresholdUp,
			double upsetThresholdLo);

    void updatePosteriorProbabilitiesInPlace(int experiment, int response, double upsetThresholdUp, double upsetThresholdLo);

    double[] calculatePosteriorProbabilities(int experiment, int response);

    void updateClassifiedAtomsAndClassifiedState(double upsetThresholdUp, double upsetThresholdLo);

    double getUpSetProbabilityMass(int state);

    boolean isClassified();

    int findHalvingStates(double prob);

    int findHalvingStatesParallel(double prob);

    double responseProbability(int experiment, int response, int trueState);
}
