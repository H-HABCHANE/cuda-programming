// matrix_mul.cu — deuxième exemple CUDA : multiplication de matrices C = A × B
// Nouveautés par rapport à vector_add.cu :
//   - grille et blocs en 2D (lignes / colonnes)
//   - mémoire partagée (__shared__) pour réutiliser les données
//   - mesure du temps sur le GPU avec cudaEvent
// Compiler : nvcc -arch=sm_75 matrix_mul.cu -o matrix_mul
// Exécuter : ./matrix_mul

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

#define TILE 16  // taille d'une tuile : blocs de 16 × 16 = 256 threads

// Version 1 (naïve) : chaque thread calcule un élément C[row][col]
// en lisant toute une ligne de A et toute une colonne de B dans la mémoire globale.
__global__ void matMulNaive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Version 2 (tuiles) : le bloc charge ensemble une tuile de A et une tuile de B
// dans la mémoire partagée (rapide), puis chaque thread les réutilise TILE fois.
// On lit ainsi beaucoup moins la mémoire globale (lente).
__global__ void matMulTiled(const float *A, const float *B, float *C, int n) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;

    // On parcourt les tuiles le long de la dimension k
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        // Chaque thread charge un élément de chaque tuile (0 si hors de la matrice)
        tileA[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[row * n + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[bRow * n + col] : 0.0f;
        __syncthreads();  // attendre que toute la tuile soit chargée

        for (int k = 0; k < TILE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();  // attendre avant d'écraser les tuiles au tour suivant
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Vérifie quelques éléments de C en les recalculant sur le CPU
int verify(const float *A, const float *B, const float *C, int n) {
    int errors = 0;
    for (int s = 0; s < 1000; s++) {
        int row = rand() % n;
        int col = rand() % n;
        float expected = 0.0f;
        for (int k = 0; k < n; k++) {
            expected += A[row * n + k] * B[k * n + col];
        }
        if (fabsf(C[row * n + col] - expected) > 1e-3f * fabsf(expected) + 1e-3f) {
            errors++;
        }
    }
    return errors;
}

int main() {
    const int n = 1024;  // matrices 1024 × 1024
    const size_t size = (size_t)n * n * sizeof(float);

    // 1. Matrices sur le CPU, remplies de valeurs aléatoires entre 0 et 1
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    for (int i = 0; i < n * n; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // 2. Matrices sur le GPU + copie CPU -> GPU
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 3. Grille 2D : un thread par élément de C
    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // Événements CUDA pour chronométrer les kernels sur le GPU
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms;

    // 4. Version naïve
    CUDA_CHECK(cudaEventRecord(start));
    matMulNaive<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    int errNaive = verify(h_A, h_B, h_C, n);
    printf("Naïve   : %7.3f ms  (%s)\n", ms, errNaive == 0 ? "correct" : "ERREUR");

    // 5. Version avec tuiles en mémoire partagée
    CUDA_CHECK(cudaMemset(d_C, 0, size));
    CUDA_CHECK(cudaEventRecord(start));
    matMulTiled<<<blocks, threads>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    int errTiled = verify(h_A, h_B, h_C, n);
    printf("Tuiles  : %7.3f ms  (%s)\n", ms, errTiled == 0 ? "correct" : "ERREUR");

    // 6. Libération
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    return (errNaive == 0 && errTiled == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
