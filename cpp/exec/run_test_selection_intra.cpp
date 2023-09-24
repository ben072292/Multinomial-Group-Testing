#include "core.hpp"
#include "product_lattice.hpp"
#include "product_lattice_dilution.hpp"
#include "product_lattice_non_dilution.hpp"

int main(int argc, char *argv[])
{

    // omp_set_num_threads(8);
    int atom = std::atoi(argv[1]);
    int variants = std::atoi(argv[2]);
    double prior = std::atof(argv[3]);

    double pi0[atom * variants];
    for (int i = 0; i < atom * variants; i++)
    {
        pi0[i] = prior;
    }

    Product_lattice *p = new Product_lattice_non_dilution(atom, variants, pi0);

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << p->halving(0.25) << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "\nTime Consumption: " << duration.count() / 1e6 << " Second." << std::endl;

    delete p;
}
