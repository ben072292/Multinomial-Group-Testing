GCC = mpicxx
ANALYSIS = statistical_analysis
TREE = tree
UTIL = util
LATTICE = product_lattice_model

all: omp debug mpi mpi_ts ts
omp:
	$(GCC) -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/global_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o gtomp

debug: 
	$(GCC) -std=c++11 -O0 -Wall -Werror -g3 -fopenmp $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/global_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o gtdebug

mpi:
	$(GCC) -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/start_mpi.cpp $(ANALYSIS)/$(TREE)/global_tree.cpp $(ANALYSIS)/$(TREE)/global_tree_mpi.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o gtmpi

mpi_ts:
	$(GCC) -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/test_selection_mpi.cpp $(ANALYSIS)/$(TREE)/global_tree.cpp $(ANALYSIS)/$(TREE)/global_tree_mpi.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o gtmpits

ts: 
	$(GCC) -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/start_test_selection.cpp $(ANALYSIS)/$(TREE)/global_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o gtts

test:
	$(GCC) -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/lattice_shrinking_test.cpp $(ANALYSIS)/$(TREE)/global_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o gttest

.PHONY: clean
clean:
	rm -rf gt*