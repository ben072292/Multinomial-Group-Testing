GCC = g++-12
OUT = a.out
all: $(OUT)
ANALYSIS = statistical_analysis
TREE = tree
UTIL = util
LATTICE = product_lattice_model

all: $(OUT)
$(OUT): $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(TREE)/single_tree_mpi.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp 	
	$(GCC) -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o $(OUT)

debug: 
	$(GCC) -std=c++11 -O0 -Wall -Werror -g3 $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o $(OUT)

mpi:
	mpicxx -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/start_mpi.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(TREE)/single_tree_mpi.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o $(OUT)

mpi_ts:
	mpicxx -std=c++11 -O3 -Wall -fopenmp $(ANALYSIS)/test_selection_mpi.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(TREE)/single_tree_mpi.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o $(OUT)
clean:
	rm $(OUT)