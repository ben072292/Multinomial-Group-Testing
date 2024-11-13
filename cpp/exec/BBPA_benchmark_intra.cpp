#include "product_lattice.hpp"

EXPORT void run_BBPA_benchmark_intra(int argc, char *argv[])
{

    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <subjs> <variants>\n";
        exit(1);
    }

    int subjs = std::atoi(argv[1]);
    int variants = std::atoi(argv[2]);

    double pi0[subjs * variants];
    for (int i = 0; i < subjs * variants; i++)
    {
        pi0[i] = 0.03;
    }

    Product_lattice *p = new Product_lattice_non_dilution(subjs, variants, pi0);
    std::cout << hardware_config_summary() << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << p->BBPA(1.0 / (1 << p->variants())) << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "\nTime Consumption: " << duration.count() / 1e6 << " Second." << std::endl;

    delete p;
}
