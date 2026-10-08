// matrix_add3.cu — troisième exemple CUDA : addition de 3 matrices D = A + B + C
// Nouveautés par rapport aux exemples précédents :
//   - matrices rectangulaires (rows × cols) dont la taille n'est pas un multiple du bloc
//   - fusion de kernels : un seul kernel au lieu de deux, moins d'accès mémoire
// Compiler : nvcc -arch=sm_75 matrix_add3.cu -o matrix_add3
// Exécuter : ./matrix_add3

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Erreur CUDA %s:%d : %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define BLOCK 16  // blocs de 16 × 16 = 256 threads

// Version 1 : addition de 2 matrices, utilisée deux fois (T = A + B, puis D = T + C)
__global__ void matAdd2(const float *X, const float *Y, float *Z, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {  // indispensable : la grille déborde de la matrice
        int i = row * cols + col;    // position de (row, col) en mémoire
        Z[i] = X[i] + Y[i];
    }
}

// Version 2 (fusionnée) : chaque thread lit A, B et C et écrit D en une seule fois
__global__ void matAdd3(const float *A, const float *B, const float *C, float *D,
                        int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int i = row * cols + col;
        D[i] = A[i] + B[i] + C[i];
    }
}

// Compare le résultat du GPU avec le calcul sur le CPU, élément par élément
int verify(const float *A, const float *B, const float *C, const float *D, int count) {
    int errors = 0;
    for (int i = 0; i < count; i++) {
        float expected = A[i] + B[i] + C[i];
        if (fabsf(D[i] - expected) > 1e-5f) errors++;
    }
    return errors;
}

int main() {
    const int rows = 2000;  // ni 2000 ni 3000 ne sont des multiples de 16
    const int cols = 3000;
    const int count = rows * cols;
    const size_t size = (size_t)count * sizeof(float);

    // 1. Matrices sur le CPU, remplies de valeurs aléatoires entre 0 et 1
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_D = (float *)malloc(size);
    for (int i = 0; i < count; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = rand() / (float)RAND_MAX;
    }

    // 2. Matrices sur le GPU (T sert de résultat intermédiaire pour la version 1)
    float *d_A, *d_B, *d_C, *d_D, *d_T;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_D, size));
    CUDA_CHECK(cudaMalloc(&d_T, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice));

    // 3. Grille 2D : blockIdx.x parcourt les colonnes, blockIdx.y les lignes
    dim3 threads(BLOCK, BLOCK);
    dim3 blocks((cols + BLOCK - 1) / BLOCK, (rows + BLOCK - 1) / BLOCK);
    printf("Matrices %d × %d, grille de %d × %d blocs\n", rows, cols, blocks.x, blocks.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // 4. Version 1 : deux kernels successifs
    CUDA_CHECK(cudaEventRecord(start));
    matAdd2<<<blocks, threads>>>(d_A, d_B, d_T, rows, cols);  // T = A + B
    matAdd2<<<blocks, threads>>>(d_T, d_C, d_D, rows, cols);  // D = T + C
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));
    int errTwo = verify(h_A, h_B, h_C, h_D, count);
    printf("2 kernels       : %7.3f ms  (%s)\n", ms, errTwo == 0 ? "correct" : "ERREUR");

    // 5. Version 2 : un seul kernel fusionné
    CUDA_CHECK(cudaMemset(d_D, 0, size));
    CUDA_CHECK(cudaEventRecord(start));
    matAdd3<<<blocks, threads>>>(d_A, d_B, d_C, d_D, rows, cols);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));
    int errFused = verify(h_A, h_B, h_C, h_D, count);
    printf("1 kernel fusionné : %7.3f ms  (%s)\n", ms, errFused == 0 ? "correct" : "ERREUR");

    // 6. Libération
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_T));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    return (errTwo == 0 && errFused == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
