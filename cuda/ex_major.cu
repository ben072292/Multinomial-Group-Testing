#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

typedef int bin_enc;

template <int _curr_subjs, int _variants, int prior_numer>
__global__ void set_prior_probs(float *_post_probs)
{
    const float pi0 = (float)(prior_numer) / 100.0;
    int s_iter = blockIdx.x * blockDim.x + threadIdx.x;
    float prob = 1.0f;
    for (int i = 0; i < _curr_subjs * _variants; i++)
    {
        if ((s_iter & (1 << i)) == 0)
            prob *= pi0;
        else
            prob *= (1.0f - pi0);
    }
    _post_probs[s_iter] = prob;
}

/** A100: N = 15, k = 2, prior = 0.1: 89.0362 seconds */
template <int _curr_subjs, int _variants>
__global__ void halving(const float *_post_probs, float *mins)
{
    int partition_id = 0;
    float partition_mass[1 << _variants];
    memset(r_mass, 0, (1 << k) * sizeof(float));
    int ex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int s_iter = 0; s_iter < (1 << (_curr_subjs * _variants)); s_iter++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < _variants; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
        // }

        partition_id |= (1 & (((ex & s_iter) - ex) >> 31));
        partition_id |= (2 & (((ex & (s_iter >> _curr_subjs)) - ex) >> 31));
        partition_mass[partition_id] += _post_probs[s_iter];
        partition_id = 0;
    }

    float min = 0.0;
    for(int i = 0; i < (1 << _variants); i++){
        min += abs(partition_mass[i] - 1.0 / (1 << _variants));
    }
    mins[ex] = min;
}

int main()
{
    float *post_probs, *mins;
    constexpr int curr_subjs = 14;
    constexpr int variants = 2;
    constexpr int prior_numer = 30;

    std::cout << "N = " << curr_subjs << ", k = " << variants << ", prior = " << (float)(prior_numer) / 100.0 << std::endl;
    int numElements = (1 << (curr_subjs * variants));
    cudaError_t cudaStatus = cudaMalloc((void **)&post_probs, numElements * sizeof(float));

    cudaStatus = cudaMalloc((void **)&mins, (1 << curr_subjs) * sizeof(float));
    // cudaStatus = cudaMemset(mins, 0, (1 << curr_subjs) * sizeof(float));

    bin_enc *d_candidate;
    cudaMalloc((void **)&d_candidate, sizeof(bin_enc));

    dim3 blockDims(256);                                          // Adjust block dimensions as needed
    dim3 gridDims((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    set_prior_probs<curr_subjs, variants, prior_numer><<<gridDims, blockDims>>>(post_probs);

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsedSeconds = end - start;

    std::cout << "Prior kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    numElements = (1 << curr_subjs);
    dim3 gridDims1((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    start = std::chrono::system_clock::now();

    halving<curr_subjs, variants><<<gridDims1, blockDims>>>(post_probs, mins);

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    end = std::chrono::system_clock::now();
    elapsedSeconds = end - start;

    std::cout << "BBPA kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    // Copy the result back from the GPU
    bin_enc h_candidate;
    cudaMemcpy(&h_candidate, d_candidate, sizeof(bin_enc), cudaMemcpyDeviceToHost);

    float *h_mins = new float[(1 << curr_subjs)];
    cudaMemcpy(h_mins, mins, (1 << curr_subjs) * sizeof(float), cudaMemcpyDeviceToHost);
    float global_min = 2.0;
    float global_candidate = -1;
    for (int i = 0; i < (1 << curr_subjs); i++)
    {
        if(h_mins[i] < global_min){
            global_min = h_mins[i];
            global_candidate = i;
        }
    }

    std::cout << "\nCandidate is: " << global_candidate << std::endl;

    // Free allocated memory on the GPU
    cudaFree(post_probs);
    cudaFree(mins);
    delete[] h_mins;

    return 0;
}