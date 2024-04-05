#!/bin/bash  
#SBATCH -N 4
#SBATCH -n 128
##SBATCH --ntasks-per-socket=16
#SBATCH --partition medium
#SBATCH --mem=0
##SBATCH --constraint=ib # uncomment this line if you need access to nodes with IB connections
#SBATCH --time=08:00:00
#SBATCH --output=output  #the output information will put into test1.qlog file
#SBATCH --job-name=bmgt
#SBATCH --export=ALL
##SBATCH --mail-user=wchen97@ucmerced.edu
##SBATCH --mail-type=ALL
#SBATCH -c 1

module purge > /dev/null 2>&1
module load openmpi/4.0.6-gcc-8.4.1

make clean
cmake .
make -j32

if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  omp_threads=$SLURM_CPUS_PER_TASK
else
  omp_threads=1
fi

export OMP_NUM_THREADS=$omp_threads


echo "Number of threads: $OMP_NUM_THREADS"
mpirun -n 128 ./ts_v1 1 0 15 2 &> v1.csv
mpirun -n 128 ./ts_v2 1 0 15 2 &> v2.csv
mpirun -n 128 ./ts_v3 1 0 15 2 &> v3.csv
mpirun -n 128 ./ts 1 0 15 2 &> v4.csv
mpirun -n 128 ./ts 1 1 15 2 &> omp.csv
