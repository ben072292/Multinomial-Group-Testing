#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

typedef int bin_enc;

template <int _curr_subjs, int _variants, int prior_numer>
__global__ void set_prior_probs(float *_post_probs)
{
    const float pi0 = (float)(prior_numer) / 100.0;
    int s_iter = blockIdx.x * blockDim.x + threadIdx.x;
    float prob = 1.0;
    for (int i = 0; i < _curr_subjs * _variants; i++)
    {
        if ((s_iter & (1 << i)) == 0)
            prob *= pi0;
        else
            prob *= (1.0 - pi0);
    }
    _post_probs[s_iter] = prob;
}

template <int _curr_subjs, int _variants>
__global__ void halving_serial_kernel_V1(const float *_post_probs, float *partition_mass, bin_enc *candidate)
{
    int partition_id = 0;
    int s_iter = blockIdx.x * blockDim.x + threadIdx.x;
    for (bin_enc experiment = 0; experiment < (1 << _curr_subjs); experiment++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < _variants; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
        // }

        partition_id |= (1 & (((experiment & s_iter) - experiment) >> 31));
        partition_id |= (2 & (((experiment & (s_iter >> _curr_subjs)) - experiment) >> 31));
        atomicAdd(&partition_mass[experiment * (1 << _variants) + partition_id], _post_probs[s_iter]);
        // partition_mass[experiment * (1 << _variants) + partition_id] += _post_probs[s_iter];
        partition_id = 0;
    }

    // __syncthreads();

    // float temp = 0.0;
    // float prob = 1.0 / (1 << _variants);
    // int experiment = blockIdx.x * blockDim.x + threadIdx.x;
    // if (experiment < (1 << _curr_subjs))
    // {
    //     for (bin_enc i = 0; i < (1 << _variants); i++)
    //     {
    //         temp += abs(partition_mass[experiment * (1 << _variants) + i] - prob);
    //     }
    //     if(temp < atomicMinFloat(&global_min, temp)){
    //         atomicCAS(&global_candidate, global_candidate, experiment);
    //     }

    // }
}

template <int _curr_subjs, int _variants>
__global__ void halving_serial_kernel(const float *_post_probs, float *partition_mass, bin_enc *candidate)
{
    volatile int partition_id = 0;
    __shared__ float block_partition_mass[(1 << _curr_subjs) * (1 << _variants)];
    int s_iter = blockIdx.x * blockDim.x + threadIdx.x;
    for (bin_enc experiment = threadIdx.x; experiment < (1 << _curr_subjs) + threadIdx.x; experiment++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < _variants; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (s_iter >> (variant * _curr_subjs))) - experiment) >> 31));
        // }
        int ex = experiment % blockDim.x;

        partition_id |= (1 & (((ex & s_iter) - ex) >> 31));
        partition_id |= (2 & (((ex & (s_iter >> _curr_subjs)) - ex) >> 31));
        block_partition_mass[ex * (1 << _variants) + partition_id] += _post_probs[s_iter];
        partition_id = 0;
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        for (int i = 0; i < (1 << _curr_subjs) * (1 << _variants); i++)
        {
            atomicAdd(&partition_mass[i], block_partition_mass[i]);
        }
    }
    __syncthreads();

    // __syncthreads();

    // float temp = 0.0;
    // float prob = 1.0 / (1 << _variants);
    // int experiment = blockIdx.x * blockDim.x + threadIdx.x;
    // if (experiment < (1 << _curr_subjs))
    // {
    //     for (bin_enc i = 0; i < (1 << _variants); i++)
    //     {
    //         temp += abs(partition_mass[experiment * (1 << _variants) + i] - prob);
    //     }
    //     if(temp < atomicMinFloat(&global_min, temp)){
    //         atomicCAS(&global_candidate, global_candidate, experiment);
    //     }

    // }
}

int main()
{
    float *post_probs, *partition_mass;
    constexpr int curr_subjs = 12;
    constexpr int variants = 2;
    constexpr int prior_numer = 1;

    std::cout << "N = " << curr_subjs << ", k = " << variants << ", prior = " << (float)(prior_numer) / 100.0 << std::endl;
    int numElements = (1 << (curr_subjs * variants));
    cudaError_t cudaStatus = cudaMalloc((void **)&post_probs, numElements * sizeof(float));
    cudaStatus = cudaMemset(post_probs, 1.0 / numElements, numElements * sizeof(float));

    cudaStatus = cudaMalloc((void **)&partition_mass, (1 << curr_subjs) * (1 << variants) * sizeof(float));
    cudaStatus = cudaMemset(partition_mass, 0.0, (1 << curr_subjs) * (1 << variants) * sizeof(float));

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

    start = std::chrono::system_clock::now();

    halving_serial_kernel_V1<curr_subjs, variants><<<gridDims, blockDims>>>(post_probs, partition_mass, d_candidate);

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    end = std::chrono::system_clock::now();
    elapsedSeconds = end - start;

    std::cout << "BBPA kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    // Copy the result back from the GPU
    bin_enc h_candidate;
    cudaMemcpy(&h_candidate, d_candidate, sizeof(bin_enc), cudaMemcpyDeviceToHost);

    float *h_partition_mass = new float[(1 << curr_subjs) * (1 << variants)];
    cudaMemcpy(h_partition_mass, partition_mass, (1 << curr_subjs) * (1 << variants) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < (1 << curr_subjs) * (1 << variants); i+=4)
    {
        std::cout << h_partition_mass[i] + h_partition_mass[i+1] + h_partition_mass[i+2] + h_partition_mass[i+3] << "  ";
    }

    std::cout << "\nCandidate is: " << h_candidate << std::endl;

    // Free allocated memory on the GPU
    cudaFree(post_probs);
    cudaFree(partition_mass);
    cudaFree(d_candidate);

    return 0;
}
