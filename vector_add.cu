// vector_add.cu — premier exemple CUDA : C = A + B
// Compiler : nvcc vector_add.cu -o vector_add
// Exécuter : ./vector_add

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Vérifie le résultat de chaque appel CUDA et arrête le programme en cas d'erreur
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Erreur CUDA %s:%d : %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel : chaque thread du GPU calcule un seul élément de C
__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // indice global du thread
    if (i < n) {                                    // évite de dépasser le tableau
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 20;  // 1 048 576 éléments
    const size_t size = n * sizeof(float);

    // 1. Allocation et initialisation sur le CPU (host)
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // 2. Allocation sur le GPU (device)
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // 3. Copie CPU -> GPU
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 4. Lancement du kernel : assez de blocs de 256 threads pour couvrir n
    const int threadsPerBlock = 256;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Copie GPU -> CPU
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 6. Vérification : chaque élément doit valoir 3.0
    int errors = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != 3.0f) errors++;
    }
    printf(errors == 0 ? "Succès : %d éléments corrects\n"
                       : "Échec : %d erreurs\n",
           errors == 0 ? n : errors);

    // 7. Libération de la mémoire
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    return errors == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
