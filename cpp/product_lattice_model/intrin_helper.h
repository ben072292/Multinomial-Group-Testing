#include "../core.hpp"

// initialize a 256bit int vector with val
FORCE_INLINE __m256i AVX2_integer(int val){return _mm256_set1_epi32(val);}

// initialize a 256bit double vector with val
FORCE_INLINE __m256d AVX2_double(double val){return _mm256_set1_pd(val);}

// 256bit bitwise right shift
FORCE_INLINE __m256i AVX2_bitwise_rshift(__m256i input, int shift_count){return _mm256_srl_epi32(input, _mm_cvtsi32_si128(shift_count));}

// 256bit bitwise and
FORCE_INLINE __m256i AVX2_bitwise_AND(__m256i a, __m256i b){return _mm256_and_si256(a, b);}

// 256bit bitwise or
FORCE_INLINE __m256i AVX2_bitwise_OR(__m256i a, __m256i b){return _mm256_or_si256(a, b);}

// 256bit int vector addition: return a+b
FORCE_INLINE __m256i AVX2_add(__m256i a, __m256i b){return _mm256_add_epi32(a, b);}

// 256bit int vector substraction: return a-b
FORCE_INLINE __m256i AVX2_substract(__m256i a, __m256i b){return _mm256_sub_epi32(a, b);}

// 256bit int vector multiplication: return a*b
FORCE_INLINE __m256i AVX2_mul(__m256i a, __m256i b){return _mm256_mul_epi32(a, b);}

FORCE_INLINE void vec_to_int(__m256i val, int out[]){
    _mm256_store_si256((__m256i *)out, val);
}

FORCE_INLINE void vec_to_double(__m256d val, double out[]){
    _mm256_store_pd(out, val);
}



// show as intger 
inline void show(__m256i val){
    alignas(32) int tmp[8];
    _mm256_store_si256((__m256i *)tmp, val);
    for(int i = 0; i < 8; i++){
        std::cout << tmp[i] << " ";
    }
    std::cout << std::endl;
}

inline void show(__m256d val){
    alignas(64) double tmp[4];
    _mm256_store_pd(tmp, val);
    for(int i = 0; i < 4; i++){
        std::cout << tmp[i] << " ";
        std::cout << std::endl;
    }
}