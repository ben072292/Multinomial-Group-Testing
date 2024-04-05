// #include <cmath>
// #include <cstdlib>
// #include <ctime>
// #include <fstream>
// #include <iostream>
// #include <vector>

// // Function to calculate the standard deviation
// double calculateSD(const std::vector<double> &numArray)
// {
//     double sum = 0.0, standardDeviation = 0.0;
//     int length = numArray.size();

//     for (double num : numArray)
//     {
//         sum += num;
//     }

//     double mean = sum / length;

//     for (double num : numArray)
//     {
//         standardDeviation += std::pow(num - mean, 2);
//     }

//     return std::sqrt(standardDeviation / length);
// }

// int main()
// {
//     std::ofstream out("new-individual.csv", std::ios::app);

//     if (!out)
//     {
//         std::cerr << "Error: Unable to open file." << std::endl;
//         return 1;
//     }

//     std::srand(static_cast<unsigned int>(std::time(nullptr)));

//     for (int m = 0; m < 4; m++)
//     {
//         for (int N = 25; N <= 25; N++)
//         {
//             int stage = 12;
//             int rounds = 10000;
//             double upsetThresholdUp = 0.005;
//             double upsetThresholdLo = 0.01;
//             double prior = 0.02;

//             if (m == 0)
//                 prior = 0.01;
//             if (m == 1)
//                 prior = 0.02;
//             if (m == 2)
//                 prior = 0.05;
//             if (m == 3)
//                 prior = 0.1;

//             std::vector<double> stages(rounds, 0.0);
//             std::vector<double> tests(rounds, 0.0);
//             std::vector<double> fp(rounds, 0.0);
//             std::vector<double> fn(rounds, 0.0);
//             std::vector<bool> classification(rounds, true);

//             double pi0[1] = {0.02};

//             for (int i = 0; i < rounds; i++)
//             {
//                 int totalPositive = 0;
//                 std::vector<int> trueStates(N);

//                 for (int j = 0; j < N; j++)
//                 {
//                     if (std::rand() / static_cast<double>(RAND_MAX) > prior)
//                     {
//                         trueStates[j] = 1;
//                         totalPositive++;
//                     }
//                     else
//                     {
//                         trueStates[j] = 0;
//                     }
//                 }

//                 std::vector<int> classificationStatus(N);

//                 for (int j = 0; j < N; j++)
//                 {
//                     if (m == 5)
//                     {
//                         if (j == 0)
//                         {
//                             prior = 0.2;
//                             pi0[0] = prior;
//                         }
//                         else
//                         {
//                             prior = 0.02;
//                             pi0[0] = prior;
//                         }
//                     }

//                     if (m == 6)
//                     {
//                         if (j == 0 || j == 1)
//                         {
//                             prior = 0.2;
//                             pi0[0] = prior;
//                         }
//                         else
//                         {
//                             prior = 0.02;
//                             pi0[0] = prior;
//                         }
//                     }

//                     if (m == 7)
//                     {
//                         if (j == 0 || j == 1 || j == 2)
//                         {
//                             prior = 0.2;
//                             pi0[0] = prior;
//                         }
//                         else
//                         {
//                             prior = 0.02;
//                             pi0[0] = prior;
//                         }
//                     }

//                     if (m == 8)
//                     {
//                         if (j == 0 || j == 1 || j == 2 || j == 3)
//                         {
//                             prior = 0.2;
//                             pi0[0] = prior;
//                         }
//                         else
//                         {
//                             prior = 0.02;
//                             pi0[0] = prior;
//                         }
//                     }

//                     PoolStat poolStat(1, pi0);
//                     LatticeDilution p(poolStat);
//                     int maxStages = 0;

//                     for (int k = 0; k < stage; k++)
//                     {
//                         int response = std::rand() / static_cast<double>(RAND_MAX) < 0.01 ? 0 : 1;
//                         int trueResponse = trueStates[j] + response == 1 ? 0 : 1;
//                         p.updatePosteriorProbability(1, trueResponse, upsetThresholdUp, upsetThresholdLo);
//                         maxStages++;

//                         if (p.isClassified())
//                         {
//                             break;
//                         }
//                     }

//                     tests[i] += maxStages;

//                     if (p.isClassified())
//                     {
//                         if (maxStages > stages[i])
//                         {
//                             stages[i] = maxStages;
//                         }

//                         if (p.getPosteriorProbabilityMap()[1] < upsetThresholdLo)
//                         {
//                             classificationStatus[j] = 0;
//                         }
//                         if (p.getPosteriorProbabilityMap()[1] > (1 - upsetThresholdUp))
//                         {
//                             classificationStatus[j] = 1;
//                         }
//                     }
//                     else
//                     {
//                         classification[i] = false;
//                         break;
//                     }
//                 }

//                 if (classification[i])
//                 {
//                     for (int j = 0; j < N; j++)
//                     {
//                         if (trueStates[j] == 1 && classificationStatus[j] == 0)
//                         {
//                             fp[i]++;
//                         }
//                         if (trueStates[j] == 0 && classificationStatus[j] == 1)
//                         {
//                             fn[i]++;
//                         }
//                     }

//                     fp[i] = totalPositive == 0 ? 0.0 : fp[i] / totalPositive;
//                     fn[i] = (N - totalPositive) == 0 ? 0.0 : fn[i] / (N - totalPositive);
//                 }
//             }

//             double totalClassified = 0;
//             double fpCount = 0;
//             double fnCount = 0;
//             double stageCount = 0.0;
//             double testCount = 0.0;

//             for (int i = 0; i < rounds; i++)
//             {
//                 if (classification[i])
//                 {
//                     totalClassified++;
//                     fpCount += fp[i];
//                     fnCount += fn[i];
//                     stageCount += stages[i];
//                     testCount += tests[i];
//                 }
//             }

//             std::vector<double> realStages;
//             std::vector<double> realTests;

//             for (int i = 0; i < rounds; i++)
//             {
//                 if (classification[i])
//                 {
//                     realStages.push_back(stages[i]);
//                     realTests.push_back(tests[i]);
//                 }
//             }

//             double stageSD = calculateSD(realStages);
//             double testSD = calculateSD(realTests);

//             out << totalClassified / rounds * 100 << "%," << fpCount / totalClassified * 100 << "%," << fnCount / totalClassified * 100 << "%," << stageCount / totalClassified << "," << testCount / totalClassified << "," << stageSD << "," << testSD << std::endl;
//         }
//     }

//     return 0;
// }
