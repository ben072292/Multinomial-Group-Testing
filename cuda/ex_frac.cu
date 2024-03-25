#include "cuda_runtime.h"
#include <stdio.h>
#include "mpi.h"
#include "nccl.h"
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <unistd.h>

#ifndef N
#define N 12
#endif

#ifndef K
#define K 2
#endif

#ifndef P
#define P 10
#endif

#ifndef B
#define B 256
#endif

#ifndef F
#define F 10
#endif

#define MPICHECK(cmd)                                \
    do                                               \
    {                                                \
        int e = cmd;                                 \
        if (e != MPI_SUCCESS)                        \
        {                                            \
            printf("Failed: MPI error %s:%d '%d'\n", \
                   __FILE__, __LINE__, e);           \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while (0)

#define CUDACHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        cudaError_t e = cmd;                                   \
        if (e != cudaSuccess)                                  \
        {                                                      \
            printf("Failed: Cuda error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, cudaGetErrorString(e)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                         \
    do                                                         \
    {                                                          \
        ncclResult_t r = cmd;                                  \
        if (r != ncclSuccess)                                  \
        {                                                      \
            printf("Failed, NCCL error %s:%d '%s'\n",          \
                   __FILE__, __LINE__, ncclGetErrorString(r)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

static uint64_t getHostHash(const char *string)
{
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++)
    {
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
}

static void getHostName(char *hostname, int maxlen)
{
    gethostname(hostname, maxlen);
    for (int i = 0; i < maxlen; i++)
    {
        if (hostname[i] == '.')
        {
            hostname[i] = '\0';
            return;
        }
    }
}

typedef int bin_enc;

__device__ bin_enc offset_to_state(int offset, int rank, int nranks){
    return (1 << (N * K)) * rank / nranks + offset;
}

template <int n, int k, int p>
__global__ void set_prior_probs(float *_post_probs, int rank, int nranks)
{
    const float pi0[30] = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f, 0.09f, 0.1f,
                           0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f, 0.17f, 0.18f, 0.19f, 0.2f,
                           0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.3f};
    int s_iter = blockIdx.x * blockDim.x + threadIdx.x;
    float prob = 1.0f;
    for (int i = 0; i < n * k; i++)
    {
        if ((offset_to_state(s_iter, rank, nranks) & (1 << i)) == 0)
            prob *= pi0[i];
        else
            prob *= (1.0f - pi0[i]);
    }
    _post_probs[s_iter] = prob;
}

/** A100: N = 15, k = 2, prior = 0.1: 46.9054 seconds
 *  RTX3060: N = 15, k = 2, prior = 0.1: 154.283 seconds
 */
template <int n, int k, int f>
__global__ void halving(const float *probs, float *mass, int rank, int nranks)
{
    float r_mass[1 << k];
    memset(r_mass, 0, (1 << k) * sizeof(float));
    int ex = (blockIdx.x * blockDim.x + threadIdx.x) % (1 << n);
    int iter = (blockIdx.x * blockDim.x + threadIdx.x) / (1 << n);
    int iters = (1 << (n * k - f)) / nranks;
    for (int s_iter = 0; s_iter < iters; s_iter++)
    {
        int state = iter * iters + s_iter;
        int partition_id = 0;
#pragma unroll k
        for (int variant = 0; variant < k; variant++)
        {
            partition_id |= ((1 << variant) & (((ex & (offset_to_state(state, rank, nranks) >> (variant * n))) - ex) >> 31));
        }
        // partition_id |= (1 & (((ex & state) - ex) >> 31));
        // partition_id |= (2 & (((ex & (state >> n)) - ex) >> 31));
        r_mass[partition_id] += probs[state];
    }

    // atomicAdd(reinterpret_cast<float4*>(mass + ex * sizeof(float4)), *reinterpret_cast<float4*>(r_mass)); // only supported starting compute capability 9.0
    for (int i = 0; i < (1 << k); i++)
    {
        atomicAdd(&mass[ex * (1 << k) + i], r_mass[i]);
        // mass[ex * (1 << k) + i] += r_mass[i];
    }
}

int main(int argc, char *argv[])
{
    int myRank, nRanks, localRank = 0;

    // initializing MPI
    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    // calculating localRank based on hostname which is used in selecting a GPU
    uint64_t hostHashs[nRanks];
    char hostname[1024];
    getHostName(hostname, 1024);
    hostHashs[myRank] = getHostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    for (int p = 0; p < nRanks; p++)
    {
        if (p == myRank)
            break;
        if (hostHashs[p] == hostHashs[myRank])
            localRank++;
    }

    // std::cout << "Rank " << myRank << "->GPU " << localRank << std::endl;

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s;
    float *d_probs, *d_mass;
    bin_enc *d_candidate;
    int numElements = (1 << (N * K)) / nRanks;
    cudaMalloc((void **)&d_candidate, sizeof(bin_enc));
    dim3 blockDims(B);                                          // Adjust block dimensions as needed
    dim3 gridDims((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    // get NCCL unique ID at rank 0 and broadcast it to all others
    if (myRank == 0)
        ncclGetUniqueId(&id);
    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // picking a GPU based on localRank, allocate device buffers
    CUDACHECK(cudaSetDevice(localRank));
    CUDACHECK(cudaMalloc((void **)&d_probs, (1 << (N * K)) * sizeof(float) / nRanks));
    CUDACHECK(cudaMalloc((void **)&d_mass, (1 << (N + K)) * sizeof(float)));
    // CUDACHECK(cudaMemset(mins, 0, (1 << curr_subjs) * sizeof(float)));
    CUDACHECK(cudaStreamCreate(&s));

    // initializing NCCL
    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    if (!myRank)
    {
        std::cout << "N = " << N << ", k = " << K << ", prior = " << P / 100.0f << std::endl;
        std::cout << "Number of GPUs: " << nRanks << std::endl;
    }

    std::chrono::time_point<std::chrono::system_clock> start, end_1, end_2, end_3;
    start = std::chrono::system_clock::now();

    set_prior_probs<N, K, P><<<gridDims, blockDims, 0, s>>>(d_probs, myRank, nRanks);

    CUDACHECK(cudaStreamSynchronize(s));

    end_1 = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsedSeconds = end_1 - start;
    if (!myRank)
        std::cout << "Prior kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    numElements = (1 << (N + F));
    dim3 gridDims1((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    halving<N, K, F><<<gridDims1, blockDims, 0, s>>>(d_probs, d_mass, myRank, nRanks);

    CUDACHECK(cudaStreamSynchronize(s));

    end_2 = std::chrono::system_clock::now();
    elapsedSeconds = end_2 - end_1;
    if (!myRank)
        std::cout << "BBPA kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    // communicating using NCCL
    NCCLCHECK(ncclAllReduce((const void *)d_mass, (void *)d_mass, (1 << (N + K)), ncclFloat, ncclSum,
                            comm, s));

    CUDACHECK(cudaStreamSynchronize(s));

    end_3 = std::chrono::system_clock::now();
    elapsedSeconds = end_3 - end_2;
    if (!myRank)
        std::cout << "BBPA kernel Allreduce time: " << elapsedSeconds.count() << " seconds" << std::endl;

    elapsedSeconds = end_3 - end_1;
    if (!myRank)
        std::cout << "BBPA kernel total execution time: " << elapsedSeconds.count() << " seconds" << std::endl;
    
    // Copy the result back from the GPU
    float *h_partition_mass = new float[10 * (1 << K)];
    CUDACHECK(cudaMemcpy(h_partition_mass, d_mass, (1 << K) * sizeof(float) * 10, cudaMemcpyDeviceToHost));

    if (!myRank)
    {
        for (int i = 0; i < 10 * (1 << K); i += (1 << K))
        {
            float total = 0.0;
            for(int j = i; j < i + (1 << K); j++){
                total += h_partition_mass[j];
            }
            std::cout << total << " ";
        }
        std::cout << std::endl;
    }

    // Free allocated memory on the GPU
    CUDACHECK(cudaFree(d_probs));
    CUDACHECK(cudaFree(d_mass));

    // Free Host
    delete[] h_partition_mass;

    // finalizing NCCL
    ncclCommDestroy(comm);

    // finalizing MPI
    MPICHECK(MPI_Finalize());

    return 0;
}
