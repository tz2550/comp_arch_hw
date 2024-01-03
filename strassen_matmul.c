#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include <math.h>

#define MATRIX_SIZE 16
#define FIXED_SEED  114514 
#define BATCH_TEST
//#define PRINT_MATRIX

uint32_t rounded_size;
uint32_t size_matrix;           // Size of matrix/vector

double** matrix_multiply_strassen(double** a, double** b, double** c, uint32_t size);
double*** subMatrices_create(double** original, uint32_t newSize);
double*** subMatrices_create_memcpy(double** original, uint32_t newSize);
void mergeSubMatrices(double*** subMatrices, double** original, uint32_t newSize);
double** matrix_allocate(uint32_t size);
double vector_multiply_double(double *a, double *b, size_t length);
double** matrix_add(double **a, double **b, double **c, uint32_t size);
double** matrix_cpy(double **a, double **c, uint32_t size);
double** matrix_add_see(double **a,double **b, double **c, uint32_t size);  //Debug function
int check_ptr_equiv(double** matrix1, double** matrix2, double** matrix3, double** matrix4);        //Debug function
double** matrix_sub(double **a, double **b, double **c, uint32_t size);
void matrix_free(double **matrix, uint32_t size);
void subMatrices_free(double ***subMatrices);
void subMatrices_free_memcpy(double ***subMatrices, uint32_t size);
double vector_multiply_raw(double *a, double *b, size_t length);

void mat_print(double** matrix, char *header, uint32_t newSize);

//double **m1, **m2, **m3, **m4, **m5, **m6, **m7;

static inline uint64_t rdtsc() 
{
    unsigned long a, d;
    asm volatile("rdtsc" : "=a"(a), "=d"(d));
    return a | ((uint64_t)d << 32);
}

int main(int argc, char *argv[])
{
    double **a, **b;              // Matrix To be Multiplied, note that B is actually B_transpose,
    double **a_r, **b_r, **c_r;   // Rounded Matrix for Strassen
    double **c, **d;              // Result matrix
    uint64_t start_time,end_time;  // Ticks
    size_t i, j, k;                // General iterator

    // Simple Input Guard, Note Upper Limit of Matmul is set to be 4096
    if (argc != 2 || sscanf(argv[1], "%u", &size_matrix) != 1 || size_matrix > 4096 || size_matrix < 16) 
    {
        size_matrix = MATRIX_SIZE;
#ifdef BATCH_TEST
        printf("Warning: Using Default Matrix Size\n\n");
#endif
    }

    printf("New Method1\r\n");

    // Malloc for Matrices
    a = matrix_allocate(size_matrix);
    b = matrix_allocate(size_matrix);
    c = matrix_allocate(size_matrix);
    d = matrix_allocate(size_matrix);

    // Random Seed
    srand(FIXED_SEED);

    // Initialize a,b with random numbers
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            //a[i][j] = ((double)rand() / RAND_MAX) * 16.0 - 8.0;
            a[i][j] = ((double) i * size_matrix + j);
            b[i][j] = ((double) i * size_matrix + j);
            //b[i][j] = ((double)rand() / RAND_MAX) * 16.0 - 8.0;
        }
    }

/* Strassen Run */
    //start_time = rdtsc();

    // Find nearest 2 power
    rounded_size = 1;
    while (rounded_size < size_matrix) 
    {
        rounded_size *= 2; 
    }

    a_r = matrix_allocate(rounded_size);
    b_r = matrix_allocate(rounded_size);
    c_r = matrix_allocate(rounded_size);

    // m1 = matrix_allocate(rounded_size);
    // m2 = matrix_allocate(rounded_size);
    // m3 = matrix_allocate(rounded_size);
    // m4 = matrix_allocate(rounded_size);
    // m5 = matrix_allocate(rounded_size);
    // m6 = matrix_allocate(rounded_size);
    // m7 = matrix_allocate(rounded_size);

    // Copy Data into Rounded Matrix
    for (i = 0; i < rounded_size; i++) 
    {
        if (i < size_matrix) 
        {
            // Copy existing data from a and b to a_r and b_r
            memcpy(a_r[i], a[i], size_matrix * sizeof(double));
            memcpy(b_r[i], b[i], size_matrix * sizeof(double));

            // Set the extra space in a_r and b_r to 0
            memset(a_r[i] + size_matrix, 0, (rounded_size - size_matrix) * sizeof(double));
            memset(b_r[i] + size_matrix, 0, (rounded_size - size_matrix) * sizeof(double));
        } 
        else 
        {
            memset(a_r[i], 0, rounded_size * sizeof(double));
            memset(b_r[i], 0, rounded_size * sizeof(double));
        }
    }

    start_time = rdtsc();
    // Master Call of Strassen, we do not check size input here
    //printf("Rounded_size:%d\r\n",rounded_size);
    matrix_multiply_strassen(a_r,b_r,c_r,rounded_size);
    end_time = rdtsc();

    // Free temporary matrices
    // matrix_free(m1, rounded_size);
    // matrix_free(m2, rounded_size);
    // matrix_free(m3, rounded_size);
    // matrix_free(m4, rounded_size);
    // matrix_free(m5, rounded_size);
    // matrix_free(m6, rounded_size);
    // matrix_free(m7, rounded_size);

    // Copy back padded result to output
    for (i = 0; i < size_matrix; i++) 
    {
        memcpy(c[i], c_r[i], size_matrix * sizeof(double));
    }
    
    //end_time = rdtsc();

#ifdef BATCH_TEST
    printf("%lu\n", (end_time - start_time));
#else
    printf("vector_multiply_strassen took %lu cycles.\n\n", (end_time - start_time));
#endif

/* Naive Run */
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

#ifdef PRINT_MATRIX
    printf("Matrix A:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%f ", a[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix B:\n");
    for (i = 0; i < size_matrix; i++)
     {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%f ", b[j][i]); // Reverse since b in code is b_transpose
        }
        printf("\n");
    }

    printf("Matrix Rouned A:\n");
    for (i = 0; i < rounded_size; i++) 
    {
        for (j = 0; j < rounded_size; j++) 
        {
            printf("%f ", a_r[i][j]);
        }
        printf("\n");
    }

    printf("\nMatrix Rouned B:\n");
    for (i = 0; i < rounded_size; i++)
     {
        for (j = 0; j < rounded_size; j++) 
        {
            printf("%f ", b_r[j][i]); // Reverse since b in code is b_transpose
        }
        printf("\n");
    }

    printf("\nResult Matrix Rounded C:\n");
    for (i = 0; i < rounded_size; i++) 
    {
        for (j = 0; j < rounded_size; j++) 
        {
            printf("%f ", c_r[i][j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix C:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%f ", c[i][j]);
        }
        printf("\n");
    }


    printf("\nResult Matrix D:\n");
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            printf("%f ", d[i][j]);
        }
        printf("\n");
    }
#endif

    //Check for correctness of calculation
    double tolerance = 1e-6;
    for (i = 0; i < size_matrix; i++) 
    {
        for (j = 0; j < size_matrix; j++) 
        {
            if (fabs(c[i][j] - d[i][j]) > tolerance) // Use fabs for absolute difference
            {
                printf("Significant difference at position (%zu, %zu): C = %f, D = %f\n", i, j, c[i][j], d[i][j]);
            }
        }
    }

    // Free All
    for (i = 0; i < size_matrix; i++)   
    {
        _mm_free(a[i]); _mm_free(b[i]); _mm_free(c[i]); _mm_free(d[i]);
    }
    _mm_free(a); _mm_free(b); _mm_free(c); _mm_free(d);

    return 0;
}

double** matrix_multiply_strassen(double** a, double** b, double** c, uint32_t size)
{
    size_t i, j, k; // General iterator

    //printf("Size:%d\r\n",size);

    // Base Case, we use intrinsics instead of rounding down all the way to single number
    if(size == 4)
    {
        for(i = 0; i<size; i++)
        {
            for(j = 0; j<size; j++)
            {
                c[i][j] = vector_multiply_double(a[i],b[j],size);
            }
        }
        return c;
    }

    // Partition Matrix
    uint32_t newSize = size / 2;
    double ***aSub = subMatrices_create(a, newSize);
    double ***bSub = subMatrices_create(b, newSize);
    double ***cSub = subMatrices_create(c, newSize);

    // Create temp variables
    double **temp1, **temp2;
    temp1 = matrix_allocate(rounded_size);
    temp2 = matrix_allocate(rounded_size);

    // Calculate M1-M7
    // Note that b here is actually b.T, 
    // so bSub[1] is acually B21.T and bSub[2] is B12.T
    double **m1 = matrix_allocate(newSize);
    matrix_add(aSub[0], aSub[3], temp1, newSize);
    matrix_add(bSub[0], bSub[3], temp2, newSize);
    matrix_multiply_strassen(temp1,temp2,m1,newSize);


    double **m2 = matrix_allocate(newSize);
    matrix_add(aSub[2], aSub[3], temp1, newSize);
    //matrix_cpy(bSub[0],temp2,newSize);
    //matrix_multiply_strassen(temp1,temp2,m2,newSize);
    matrix_multiply_strassen(temp1,bSub[0],m2,newSize);

    double **m3 = matrix_allocate(newSize);
    //matrix_cpy(aSub[0],temp1,newSize);
    matrix_sub(bSub[2], bSub[3], temp2, newSize);
    //matrix_multiply_strassen(temp1,temp2,m3,newSize);
    matrix_multiply_strassen(aSub[0],temp2,m3,newSize);

    double **m4 = matrix_allocate(newSize);
    //matrix_cpy(aSub[3],temp1,newSize);
    matrix_sub(bSub[1], bSub[0], temp2, newSize);
    //matrix_multiply_strassen(temp1,temp2, m4,newSize);
    matrix_multiply_strassen(aSub[3],temp2, m4,newSize);

    double **m5 = matrix_allocate(newSize);
    matrix_add(aSub[0], aSub[1], temp1, newSize);
    //matrix_cpy(bSub[3], temp2,newSize);
    //matrix_multiply_strassen(temp1,temp2,m5,newSize);
    matrix_multiply_strassen(temp1,bSub[3],m5,newSize);

    double **m6 = matrix_allocate(newSize);
    matrix_sub(aSub[2], aSub[0], temp1, newSize);
    matrix_add(bSub[0], bSub[2], temp2, newSize);
    matrix_multiply_strassen(temp1,temp2,m6,newSize);

    double **m7 = matrix_allocate(newSize);
    matrix_sub(aSub[1], aSub[3], temp1, newSize);
    matrix_add(bSub[1], bSub[3], temp2, newSize);
    matrix_multiply_strassen(temp1,temp2,m7,newSize);

    // C_11
    matrix_add(m1, m4, temp1, newSize);
    matrix_sub(temp1, m5, temp2, newSize);
    matrix_add(temp2, m7, cSub[0], newSize);

    // C_12
    matrix_add(m3, m5, cSub[1], newSize);

    // C_21
    matrix_add(m2, m4, cSub[2], newSize);

    // C_22
    matrix_sub(m1, m2, temp1, newSize);
    matrix_add(temp1, m3, temp2, newSize);
    matrix_add(temp2, m6, cSub[3], newSize);

    // Free temporary matrices 
    matrix_free(m1, newSize);
    matrix_free(m2, newSize);
    matrix_free(m3, newSize);
    matrix_free(m4, newSize);
    matrix_free(m5, newSize);
    matrix_free(m6, newSize);
    matrix_free(m7, newSize);
    matrix_free(temp1, rounded_size);
    matrix_free(temp2, rounded_size);

    // Free partitioned sub-matrices of a, b, and c
    //subMatrices_free_memcpy(aSub, newSize);
    //subMatrices_free_memcpy(bSub, newSize);
    subMatrices_free(aSub);
    subMatrices_free(bSub);
    subMatrices_free(cSub);

    return c;
}

void mat_print(double** matrix, char *header, uint32_t newSize)
{
    printf("%s\r\n", header);
    for (int i = 0; i < newSize; i++) 
    {
        for (int j = 0; j < newSize; j++) 
        {
            printf("%f ", matrix[i][j]);
        }
        printf("\n");
    } 
}

/* direct assignment */
double*** subMatrices_create(double** original, uint32_t newSize) 
{
    double*** subMatrices = (double***)_mm_malloc(4 * sizeof(double**),32);
    if (subMatrices == NULL) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    // Assigning pointers to the appropriate locations in the original matrix
    subMatrices[0] = (double**)_mm_malloc(newSize * sizeof(double*),32);        // Top-left
    subMatrices[1] = (double**)_mm_malloc(newSize * sizeof(double*),32);        // Top-right
    subMatrices[2] = (double**)_mm_malloc(newSize * sizeof(double*),32);        // Bottom-left
    subMatrices[3] = (double**)_mm_malloc(newSize * sizeof(double*),32);        // Bottom-right

    for (uint32_t i = 0; i < newSize; i++) 
    {
        subMatrices[0][i] = original[i] + 0;
        subMatrices[1][i] = original[i] + newSize;
        subMatrices[2][i] = original[i + newSize];
        subMatrices[3][i] = original[i + newSize] + newSize;
    }

    return subMatrices;
}

/* memcpy */
double*** subMatrices_create_memcpy(double** original, uint32_t newSize) 
{
    double*** subMatrices = (double***)_mm_malloc(4 * sizeof(double**), 32);
    if (subMatrices == NULL) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 4; i++) 
    {
        subMatrices[i] = matrix_allocate(newSize);
    }

    for (uint32_t i = 0; i < newSize; i++) 
    {
        // Copying rows for top-left and top-right submatrices
        memcpy(subMatrices[0][i], &original[i][0], newSize * sizeof(double)); // Top-left
        memcpy(subMatrices[1][i], &original[i][newSize], newSize * sizeof(double)); // Top-right

        // Copying rows for bottom-left and bottom-right submatrices
        memcpy(subMatrices[2][i], &original[i + newSize][0], newSize * sizeof(double)); // Bottom-left
        memcpy(subMatrices[3][i], &original[i + newSize][newSize], newSize * sizeof(double)); // Bottom-right
    }

    return subMatrices;
}

void mergeSubMatrices(double*** subMatrices, double** original, uint32_t newSize) 
{
    for (uint32_t i = 0; i < newSize; i++) 
    {
        // Merging rows for top-left and top-right submatrices
        memcpy(&original[i][0], subMatrices[0][i], newSize * sizeof(double)); // Top-left
        memcpy(&original[i][newSize], subMatrices[1][i], newSize * sizeof(double)); // Top-right

        // Merging rows for bottom-left and bottom-right submatrices
        memcpy(&original[i + newSize][0], subMatrices[2][i], newSize * sizeof(double)); // Bottom-left
        memcpy(&original[i + newSize][newSize], subMatrices[3][i], newSize * sizeof(double)); // Bottom-right
    }
}

double** matrix_allocate(uint32_t size) 
{
    double** matrix = (double **)_mm_malloc(size * sizeof(double *), 32);
    if (matrix == NULL) 
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    for (uint32_t i = 0; i < size; i++) 
    {
        matrix[i] = (double *)_mm_malloc(size * sizeof(double), 32);
        if (matrix[i] == NULL) 
        {
            fprintf(stderr, "Memory allocation failed\n");
            for (uint32_t j = 0; j < i; j++) 
            {
                free(matrix[j]);
            }
            free(matrix);
            exit(EXIT_FAILURE);
        }
    }
    return matrix;
}

double** matrix_cpy(double **a, double **c, uint32_t size) 
{
    for (uint32_t i = 0; i < size; i++) 
    {
        // Copy each row of matrix 'a' to matrix 'c'
        memcpy(c[i], a[i], size * sizeof(double));
    }

    return c;
}

double** matrix_add(double **a,double **b, double **c, uint32_t size)
{
    for (uint32_t i = 0; i < size; i++) 
    {
        for (uint32_t j = 0; j < (size & ~(size_t)3); j += 4) 
        {
            __m256d a_vec = _mm256_load_pd(&a[i][j]);
            __m256d b_vec = _mm256_load_pd(&b[i][j]);
            __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
            _mm256_store_pd(&c[i][j], result_vec);
        }
    }

    return c;
}

double** matrix_add_see(double **a,double **b, double **c, uint32_t size) 
{   
    mat_print((double**)a, "Mat_add_A_pre", size);
    mat_print((double**)c, "Mat_add_C_pre", rounded_size);
    for (uint32_t i = 0; i < size; i++) 
    {
        mat_print((double**)a, "Mat_add_A_middle_pre", size);
        for (uint32_t j = 0; j < (size & ~(size_t)3); j += 4) 
        {
            __m256d a_vec = _mm256_load_pd(&a[i][j]);
            __m256d b_vec = _mm256_load_pd(&b[i][j]);
            __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
            mat_print((double**)a, "Mat_add_A_middle_4", size);
            mat_print((double**)c, "Mat_add_C_middle_4", rounded_size);
            _mm256_storeu_pd(&c[i][j], result_vec);
            mat_print((double**)a, "Mat_add_A_middle_5", size);
            mat_print((double**)c, "Mat_add_C_middle_5", rounded_size);
        }
        mat_print((double**)a, "Mat_add_A_middle_after", size);
        mat_print((double**)c, "Mat_add_C_after", rounded_size);
    }
    mat_print((double**)a, "Mat_add_A_final", size);
    mat_print((double**)c, "Mat_add_C_final", rounded_size);

    return c;
}

double** matrix_sub(double **a,double **b, double **c, uint32_t size) 
{
    for (uint32_t i = 0; i < size; i++) 
    {
        for (uint32_t j = 0; j < (size & ~(size_t)3); j += 4) 
        {
            __m256d a_vec = _mm256_load_pd(&a[i][j]);
            __m256d b_vec = _mm256_load_pd(&b[i][j]);
            __m256d result_vec = _mm256_sub_pd(a_vec, b_vec);
            _mm256_store_pd(&c[i][j], result_vec);
        }
    }

    return c;
}

double vector_multiply_double(double *a, double *b, size_t length) 
{
    __m256d sum = _mm256_setzero_pd();
    size_t cursor = 0;

    for ( ; cursor < (length & ~(size_t)3); cursor += 4) 
    {
        __m256d a_vec256 = _mm256_load_pd(&a[cursor]);  // Load next 4 doubles from a
        __m256d b_vec256 = _mm256_load_pd(&b[cursor]);  // Load next 4 doubles from b
        __m256d result_vec256 = _mm256_mul_pd(a_vec256, b_vec256); // Multiply pairs of doubles
        sum = _mm256_add_pd(sum, result_vec256);        // Add to the running total
    }
    
    // Dealing with the leftover of length % 4, should not happen in Strassen Scenario
    if(cursor < length)
    {
        fprintf(stderr, "Vector_Mul_Double failed\n");
        exit(EXIT_FAILURE);
    }

    double final_sum[4];
    _mm256_store_pd(final_sum, sum);
    return final_sum[0] + final_sum[1] + final_sum[2] + final_sum[3];
}


void matrix_free(double **matrix, uint32_t size) 
{
    if (matrix != NULL) 
    {
        for (uint32_t i = 0; i < size; i++) 
        {
            if (matrix[i] != NULL) 
            {
                _mm_free(matrix[i]);
            }
        }
        _mm_free(matrix);
    }
}

void subMatrices_free(double*** subMatrices) 
{
    if (subMatrices != NULL) 
    {
        // Free each submatrix
        for (int i = 0; i < 4; i++) 
        {
            if (subMatrices[i] != NULL) 
            {
                _mm_free(subMatrices[i]);
            }
        }

        // Free the array of submatrices
        _mm_free(subMatrices);
    }
}

void subMatrices_free_memcpy(double ***subMatrices, uint32_t size) 
{
    if (subMatrices != NULL) 
    {
        for (int i = 0; i < 4; i++) 
        {
            if (subMatrices[i] != NULL) 
            {
                matrix_free(subMatrices[i], size);
            }
        }
        _mm_free(subMatrices);
    }
}

double vector_multiply_raw(double *a, double *b, size_t length) 
{
    double result = 0;
    for (int i = 0; i < length; i++) 
    {
        result += a[i] * b[i];
    }
    return result;
}

int check_ptr_equiv(double** matrix1, double** matrix2, double** matrix3, double** matrix4) 
{
    // Check if matrix1 is equal to any of the other matrices
    if (matrix1 == matrix2) return 1;
    if (matrix1 == matrix3) return 2;
    if (matrix1 == matrix4) return 3;

    // Check if matrix2 is equal to any of the remaining matrices
    if (matrix2 == matrix3) return 4;
    if (matrix2 == matrix4) return 5;

    // Check if matrix3 is equal to matrix4
    if (matrix3 == matrix4) return 6;

    // If none are equal
    return 0;
}