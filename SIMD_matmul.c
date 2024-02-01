#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>

#define MATRIX_SIZE 8
#define FIXED_SEED  114514  // Fix random seed for reproducibility
//#define PRINT_MATRIX
#define BATCH_TEST
#define MATRIX_TEST

static inline uint64_t rdtsc() 
{
    unsigned long a, d;
    asm volatile("rdtsc" : "=a"(a), "=d"(d));
    return a | ((uint64_t)d << 32);
}

int32_t vector_multiply_raw(int32_t *a, int32_t *b, size_t length) 
{
    int32_t result = 0;
    for (int i = 0; i < length; i++) 
    {
        result += a[i] * b[i];
    }
    return result;
}

// Not used in submission as performance improvement is not as great
// 64 bit for overflow guard
int64_t vector_multiply_64bit(int32_t *a, int32_t *b, size_t length) 
{
    __m256i sum = _mm256_setzero_si256();
    size_t cursor = 0;

    // Loop until the largest multiple of 4
    for (; cursor < (length & ~(size_t)3); cursor += 4) 
    {
        __m128i a_vec128 = _mm_loadu_si128((__m128i*)&a[cursor]);
        __m128i b_vec128 = _mm_loadu_si128((__m128i*)&b[cursor]);

        // Convert into uint64_t
        __m256i a_vec256 = _mm256_cvtepi32_epi64(a_vec128);
        __m256i b_vec256 = _mm256_cvtepi32_epi64(b_vec128);

        // Actual Multiply
        __m256i result_vec256 = _mm256_mul_epi32(a_vec256, b_vec256);

        // Store the result
        sum = _mm256_add_epi64(sum,result_vec256);
    }

    // Individual Handling of the rest
    if(cursor < length)
    {
        int32_t tail_a[4] = {0, 0, 0, 0};
        int32_t tail_b[4] = {0, 0, 0, 0};

        memcpy(tail_a, &a[cursor], (length & (size_t)3) * sizeof(int32_t)); 
        memcpy(tail_b, &b[cursor], (length & (size_t)3) * sizeof(int32_t)); 

        // Same operations here
        __m128i a_vec128 = _mm_loadu_si128((__m128i*)tail_a);
        __m128i b_vec128 = _mm_loadu_si128((__m128i*)tail_b);

        __m256i a_vec256 = _mm256_cvtepi32_epi64(a_vec128);
        __m256i b_vec256 = _mm256_cvtepi32_epi64(b_vec128);

        // Multiply
        __m256i result_vec256 = _mm256_mul_epi32(a_vec256, b_vec256);

        // Add to the sum vector
        sum = _mm256_add_epi64(sum, result_vec256);
    }
    
    int64_t result[4];
    _mm256_storeu_si256((__m256i*)result, sum);                       
    int64_t final_sum = result[0] + result[1] + result[2] + result[3];

    return final_sum;
}

// (2.3-1)
int32_t vector_multiply_32bit(int32_t *a, int32_t *b, size_t length) 
{
    __m256i sum = _mm256_setzero_si256();
    int32_t cursor = 0;

    for ( ;cursor < (length & ~(size_t)7); cursor += 8) 
    {
        __m256i a_vec256 = _mm256_load_si256((__m256i*)&a[cursor]);     // Load next 8 integers from a
        __m256i b_vec256 = _mm256_load_si256((__m256i*)&b[cursor]);     // Load next 8 integers from b
        __m256i result_vec256 = _mm256_mullo_epi32(a_vec256, b_vec256); // Multiply pairs of integers
        sum = _mm256_add_epi32(sum, result_vec256);                     // Add to the running total
    }
    
    // Dealing leftover of length % 8
    if(cursor < length)
    {
        // Add Padding
        __attribute__((aligned(32))) int32_t tail_a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        __attribute__((aligned(32))) int32_t tail_b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        memcpy(tail_a, &a[cursor], (length & (size_t)7) * sizeof(int32_t)); 
        memcpy(tail_b, &b[cursor], (length & (size_t)7) * sizeof(int32_t)); 
        
        // Same as above
        __m256i a_vec256 = _mm256_load_si256((__m256i*)tail_a);
        __m256i b_vec256 = _mm256_load_si256((__m256i*)tail_b);
        __m256i result_vec256 = _mm256_mullo_epi32(a_vec256, b_vec256);
        sum = _mm256_add_epi32(sum, result_vec256);
    }

    // Horizontal sum, see intel doc
    sum = _mm256_hadd_epi32(sum, sum);
    sum = _mm256_hadd_epi32(sum, sum);

    // Now the sum[0] and sum[4] contain the sums of the first and second 128-bit sectors in the m256i
    return _mm256_extract_epi32(sum, 0) + _mm256_extract_epi32(sum, 4);
}

// Merged for submission only, enable MATRIX_TEST for testing random matrix multiplication
#ifdef MATRIX_TEST
int main(int argc, char *argv[]) 
{
    int32_t **a, **b;              // Matrix To be Multiplied, note that B is actually B_transpose,
    int32_t **c, **d;              // Result matrix
    uint64_t start_time,end_time;  // Ticks
    size_t i, j, k;                // General iterator
    int32_t size_matrix;           // Size of matrix/vector

    // Simple Input Guard
    if (argc != 2 || sscanf(argv[1], "%d", &size_matrix) != 1) 
    {
        size_matrix = MATRIX_SIZE;
#ifdef BATCH_TEST
        printf("Warning: Using Default Matrix Size\n\n");
#endif
    }

    // Malloc
    a = (int32_t **)_mm_malloc(size_matrix * sizeof(int32_t *), 32);
    b = (int32_t **)_mm_malloc(size_matrix * sizeof(int32_t *), 32);
    c = (int32_t **)_mm_malloc(size_matrix * sizeof(int32_t *), 32);
    d = (int32_t **)_mm_malloc(size_matrix * sizeof(int32_t *), 32);
    if (!a || !b || !c || !d) 
    {
        printf("Memory allocation failed,exiting\n");
        _mm_free(a); _mm_free(b); _mm_free(c); _mm_free(d);
        exit(1);
    }

    for (i = 0; i < size_matrix; i++) 
    {
        a[i] = (int32_t *)_mm_malloc(size_matrix * sizeof(int32_t), 32);
        b[i] = (int32_t *)_mm_malloc(size_matrix * sizeof(int32_t), 32);
        c[i] = (int32_t *)_mm_malloc(size_matrix * sizeof(int32_t), 32);
        d[i] = (int32_t *)_mm_malloc(size_matrix * sizeof(int32_t), 32);

        if (!a[i] || !b[i] || !c[i] || !d[i]) 
        {
            printf("Memory allocation failed for row %zu, exiting\n", i);
            for (size_t j = 0; j < i; j++) 
            {
                _mm_free(a[j]); _mm_free(b[j]); _mm_free(c[j]); _mm_free(d[j]);
            }
            _mm_free(a); _mm_free(b); _mm_free(c); _mm_free(d);
            exit(1);
        }
    }

    // Random Seed
    srand(FIXED_SEED);

    // Initialize a,b with random numbers
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            a[i][j] = rand() % 511 - 255;
            b[i][j] = rand() % 511 - 255;
        }
    }

    // SIMD Run
    start_time = rdtsc();
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            c[i][j] = vector_multiply_32bit(a[i], b[j], size_matrix);
        }
    }
    end_time = rdtsc();
#ifdef BATCH_TEST
    printf("%lu,", (end_time - start_time));
#else
    printf("vector_multiply_32bit took %lu cycles.\n\n", (end_time - start_time));
#endif

    // Naive Run
    start_time = rdtsc();
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            d[i][j] = vector_multiply_raw(a[i], b[j], size_matrix);
        }
    }
    end_time = rdtsc();
#ifdef BATCH_TEST
    printf("%lu\n", (end_time - start_time));
#else
    printf("vector_multiply_raw took %lu cycles.\n\n", (end_time - start_time));
#endif

    // Print result matrix
#ifdef PRINT_MATRIX
    printf("Matrix A:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%d ", a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (i = 0; i < size_matrix; i++)
     {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%d ", b[j][i]); // Reverse since b in code is b_transpose
        }
        printf("\n");
    }

    printf("\nResult Matrix C:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%d ", c[i][j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix D:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%d ", d[i][j]);
        }
        printf("\n");
    }
#endif
    // Free All
    for (i = 0; i < size_matrix; i++)   
    {
        _mm_free(a[i]); _mm_free(b[i]); _mm_free(c[i]); _mm_free(d[i]);
    }
    _mm_free(a); _mm_free(b); _mm_free(c); _mm_free(d);

    return 0;
}
#else
int main(int argc, char *argv[]) 
{
    int32_t *a, *b;                // Vectors To be Multiplied, note that B is actually B_transpose,
    int32_t c, d;                  // Result number
    uint64_t start_time,end_time;  // Ticks
    size_t i, j, k;                // General iterator
    int32_t size_matrix;           // Size of matrix/vector

    // Simple Input Guard
    if (argc != 2 || sscanf(argv[1], "%d", &size_matrix) != 1) 
    {
        size_matrix = MATRIX_SIZE;
#ifdef BATCH_TEST
        printf("Warning: Using Default Matrix Size\n\n");
#endif
    }

    // Malloc
    a = (int32_t *)_mm_malloc(size_matrix * sizeof(int32_t), 32);
    b = (int32_t *)_mm_malloc(size_matrix * sizeof(int32_t), 32);
    if (!a || !b) 
    {
        printf("Memory allocation failed,exiting\n");
        _mm_free(a); _mm_free(b);
        exit(1);
    }

    // Random Seed
    srand(FIXED_SEED);

    // Initialize a,b with random numbers
    for (i = 0; i < size_matrix; i++) 
    {
        a[i] = rand() % 511 - 255;
        b[i] = rand() % 511 - 255;
    }

    // SIMD Run
    start_time = rdtsc();
    c = vector_multiply_32bit(a, b, size_matrix);
    end_time = rdtsc();
#ifdef BATCH_TEST
    printf("%lu,", (end_time - start_time));
#else
    printf("vector_multiply_32bit took %lu cycles.\n\n", (end_time - start_time));
#endif

    // Naive Run
    start_time = rdtsc();
    d = vector_multiply_raw(a, b, size_matrix);
    end_time = rdtsc();
#ifdef BATCH_TEST
    printf("%lu\n", (end_time - start_time));
#else
    printf("vector_multiply_raw took %lu cycles.\n\n", (end_time - start_time));
#endif

    if( c!= d)
    {
        printf("Warning: result from both approach differs\n");
    }

    // Print result matrix
#ifdef PRINT_MATRIX
    printf("Vector A:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        printf("%d ", a[i]);
        printf("\n");
    }

    printf("Vector B:\n");
    for (i = 0; i < size_matrix; i++)
     {
        printf("%d ", b[i]);
        printf("\n");
    }

    printf("\nResult Sum C, from intrinsics:\n");
    printf("%d ", c);
    printf("\n");

    printf("\nResult Sum D, from naive implementation:\n");
    printf("%d ", d);
    printf("\n");
#endif
    // Free All
    _mm_free(a); _mm_free(b);

    return 0;
}
#endif