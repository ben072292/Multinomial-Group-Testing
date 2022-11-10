GCC = g++
OUT = a.out
all: $(OUT)
ANALYSIS = statistical_analysis
TREE = tree
UTIL = util
LATTICE = product_lattice_model

$(OUT): $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp 	
	$(GCC) -std=c++11 -Wall -g $(ANALYSIS)/start.cpp $(ANALYSIS)/$(TREE)/single_tree.cpp $(ANALYSIS)/$(UTIL)/tree_stat.cpp $(LATTICE)/product_lattice.cpp $(LATTICE)/product_lattice_dilution.cpp $(LATTICE)/product_lattice_non_dilution.cpp -o $(OUT)

clean:
	rm $(OUT)