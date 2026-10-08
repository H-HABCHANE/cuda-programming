// reduction.cu — quatrième exemple CUDA : somme de tous les éléments d'un tableau
// Nouveautés par rapport aux exemples précédents :
//   - réduction parallèle : beaucoup de valeurs combinées en une seule
//   - opérations atomiques (atomicAdd) pour écrire sans conflit au même endroit
//   - réduction en arbre dans la mémoire partagée
//   - échanges entre threads d'un même warp (__shfl_down_sync)
// Compiler : nvcc -arch=sm_75 reduction.cu -o reduction
// Exécuter : ./reduction

#include <cstdio>
#include <cstdlib>
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

#define BLOCK 256  // threads par bloc (doit être une puissance de 2)

// Version 1 (atomique) : chaque thread ajoute sa valeur directement au total.
// Simple, mais des millions de threads se bousculent sur la même case mémoire.
__global__ void reduceAtomic(const int *in, int *sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(sum, in[i]);
    }
}

// Version 2 (arbre en mémoire partagée) : le bloc additionne ses 256 valeurs
// deux par deux (256 -> 128 -> 64 -> ... -> 1), puis un seul thread
// ajoute le résultat du bloc au total : 256 fois moins d'atomicAdd.
__global__ void reduceShared(const int *in, int *sum, int n) {
    __shared__ int cache[BLOCK];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    cache[tid] = (i < n) ? in[i] : 0;  // 0 ne change pas la somme
    __syncthreads();

    // À chaque tour, la première moitié des threads ajoute la seconde moitié
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cache[tid] += cache[tid + s];
        }
        __syncthreads();  // attendre la fin du tour avant le suivant
    }

    if (tid == 0) {
        atomicAdd(sum, cache[0]);
    }
}

// Version 3 (warp) : un warp = 32 threads qui avancent ensemble.
// __shfl_down_sync permet de lire la variable d'un autre thread du warp
// sans passer par la mémoire : 32 -> 16 -> 8 -> 4 -> 2 -> 1.
__global__ void reduceWarp(const int *in, int *sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int value = (i < n) ? in[i] : 0;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    if ((threadIdx.x % 32) == 0) {  // le premier thread de chaque warp
        atomicAdd(sum, value);
    }
}

// Lance une version, la chronomètre et vérifie le résultat
typedef void (*ReduceKernel)(const int *, int *, int);

bool run(const char *name, ReduceKernel kernel, const int *d_in, int *d_sum,
         int n, long long expected) {
    int blocks = (n + BLOCK - 1) / BLOCK;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(int)));  // remettre le total à 0
    CUDA_CHECK(cudaEventRecord(start));
    kernel<<<blocks, BLOCK>>>(d_in, d_sum, n);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    int result;
    CUDA_CHECK(cudaMemcpy(&result, d_sum, sizeof(int), cudaMemcpyDeviceToHost));
    bool ok = (result == expected);
    printf("%-20s : %7.3f ms  somme = %d  (%s)\n", name, ms, result,
           ok ? "correct" : "ERREUR");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ok;
}

int main() {
    const int n = 1 << 24;  // 16 777 216 éléments
    const size_t size = n * sizeof(int);

    // 1. Tableau d'entiers entre 0 et 9, et somme attendue calculée sur le CPU
    //    (entiers : le résultat doit être exactement le même que sur le CPU)
    int *h_in = (int *)malloc(size);
    long long expected = 0;
    for (int i = 0; i < n; i++) {
        h_in[i] = rand() % 10;
        expected += h_in[i];
    }
    printf("%d éléments, somme attendue = %lld\n", n, expected);

    // 2. Copie sur le GPU
    int *d_in, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_in, size));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // 3. Les trois versions, de la plus simple à la plus rapide
    bool ok = true;
    ok &= run("1. atomicAdd", reduceAtomic, d_in, d_sum, n, expected);
    ok &= run("2. mémoire partagée", reduceShared, d_in, d_sum, n, expected);
    ok &= run("3. warp shuffle", reduceWarp, d_in, d_sum, n, expected);

    // 4. Libération
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_sum));
    free(h_in);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
