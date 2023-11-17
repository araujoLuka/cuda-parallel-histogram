#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "cuda.h"

// GTX 1080 Ti - nv00 - Constant Memory
constexpr static uint8_t MP = 28;
constexpr static uint16_t MAX_THREADS_PER_BLOCK = 1024;
constexpr static uint32_t MAX_SHARED_MEM_PER_BLOCK = 49152;
constexpr static uint32_t MAX_SHARED_MEM_PER_MP = 98304;
// Adjustable parameters
constexpr static uint16_t THREADS_PER_BLOCK = 1024;
constexpr static uint32_t H_MAX = MAX_SHARED_MEM_PER_BLOCK / 8;
// Other constants
constexpr static uint8_t MAX_PRINT = 10;

__host__ __device__ constexpr float LARGURA_FAIXA(float nMin, float nMax,
                                                  int h) {
    return (nMax - nMin) / h;
}

__global__ void blockHisto(unsigned int *HH, const int h, const float *Input,
                           const int nTotalElements, const float nMin,
                           const float nMax) {
    //   cada bloco de threads deve ter um histograma local
    __shared__ unsigned int histoPrivate[H_MAX];

#ifdef DEBUG
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("DEBUG: threadIdx.x: %d\n", threadIdx.x);
        printf("DEBUG: blockIdx.x: %d\n", blockIdx.x);
        printf("DEBUG: blockDim.x: %d\n", blockDim.x);
        printf("DEBUG: gridDim.x: %d\n", gridDim.x);
    }
#endif
    
    //  inicializa o histograma local com zeros
    if (threadIdx.x < h) {
        histoPrivate[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (i < nTotalElements) {
        //  calcula a faixa do elemento i
        int faixa = (Input[i] - nMin) / LARGURA_FAIXA(nMin, nMax, h);
        //  incrementa o contador do histograma local
        atomicAdd(&histoPrivate[faixa], 1);
        i += stride;
    }

    // sincroniza as threads do bloco
    __syncthreads();

    // copia o histograma local com uma linha da matriz HH
    // total de linhas da matriz HH = NUM_BLOCOS
    if (threadIdx.x < gridDim.x) {
        for (int i = 0; i < h; ++i) {
            HH[blockIdx.x * h + i] = histoPrivate[i];
        }
    }
}

__global__ void globalHisto(unsigned int *H, const int h, const float *Input,
                            const int nTotalElements, const float nMin,
                            const float nMax) {
    //   cada bloco de threads deve ter um histograma local
    __shared__ unsigned int histoPrivate[H_MAX];

    //  inicializa o histograma local com zeros
    if (threadIdx.x < h) {
        histoPrivate[threadIdx.x] = 0;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (i < nTotalElements) {
        //  calcula a faixa do elemento i
        int faixa = (Input[i] - nMin) / LARGURA_FAIXA(nMin, nMax, h);
        //  incrementa o contador do histograma local
        atomicAdd(&histoPrivate[faixa], 1);
        i += stride;
    }

    // sincroniza as threads do bloco
    __syncthreads();

    //  faz o merge dos histogramas locais em um histograma global
    if (threadIdx.x < h) {
        atomicAdd(&H[threadIdx.x], histoPrivate[threadIdx.x]);
    }
}

__host__ unsigned int *serialHisto(const float *Input, const int nTotalElements,
                                   const float nMin, const float nMax,
                                   const int h) {
    unsigned int *H{new unsigned int[h]};
    for (int i = 0; i < h; ++i) {
        H[i] = 0;
    }
    for (int i = 0; i < nTotalElements; ++i) {
        int faixa = (Input[i] - nMin) / LARGURA_FAIXA(nMin, nMax, h);
        H[faixa]++;
    }

#ifdef DEBUG
    // debug print
    for (int i{0}; i < 5; ++i) {
        std::cout << "DEBUG: H[" << i << "]: " << H[i] << std::endl;
    }
#endif

    return H;
}

__global__ void validaResultados(const unsigned int *const H,
                                 const unsigned int *const HH, const int h, const int nb,
                                 const unsigned int *const S, bool *result) {
    __shared__ unsigned int BH[H_MAX];

    // soma os histogramas locais de blockHisto
    for (int i = 0; i < h; ++i) {
        BH[i] = 0;
    }
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < h; ++j) {
            BH[j] += HH[i * h + j];
        }
    }

    // printa a matriz HH
    printf("Resultado do kernel blockHisto (linhas):\n");
    for (int i = 0; i < 5; ++i) {
        printf("linha  %d: ", i+1);
        for (int j = 0; j < MAX_PRINT; ++j) {
            printf("%d ", HH[i * h + j]);
        }
        printf("...\n");
    }
    printf("...\n");
    printf("linha nb: ");
    for (int j = 0; j < MAX_PRINT; ++j) {
        printf("%d ", HH[(nb - 1) * h + j]);
    }
    printf("...\n\n");

    // printa o vetor BH
    printf("Resultado do kernel blockHisto:\n");
    for (int i = 0; i < MAX_PRINT; ++i) {
        printf("%d ", BH[i]);
    }
    printf("...\n\n");

    // printa o vetor H
    printf("Resultado do kernel globalHisto:\n");
    for (int i = 0; i < MAX_PRINT; ++i) {
        printf("%d ", H[i]);
    }
    printf("...\n\n");

    // printa o vetor S
    printf("Resultado da versao serial:\n");
    for (int i = 0; i < MAX_PRINT; ++i) {
        printf("%d ", S[i]);
    }
    printf("...\n\n");

    // compara globalHisto com serialHisto
    for (int i = 0; i < h; ++i) {
        if (H[i] != S[i]) {
            *result = false;
            printf(
                "Falha: Resultado de globalHisto e serialHisto sao "
                "diferentes!\n");
            printf("Erro: H[%d] = %d != S[%d] = %d\n", i, H[i], i, S[i]);
            return;
        }
    }
    printf(
        "Sucesso: Resultados de globalHisto e serialHisto sao "
        "iguais!\n");

    // compara blockHisto com serialHisto
    for (int i = 0; i < h; ++i) {
        if (BH[i] != S[i]) {
            *result = false;
            printf(
                "Falha: Resultado de blockHisto e serialHisto sao "
                "diferentes!\n");
            printf("Erro: BH[%d] = %d != S[%d] = %d\n", i, BH[i], i, S[i]);
            return;
        }
    }
    printf(
        "Sucesso: Resultados de blockHisto e serialHisto sao "
        "iguais!\n");

    *result = true;
    printf(
        "Sucesso: Resultados de blockHisto, globalHisto e serialHisto sao "
        "iguais!\n");
}

int main(int argc, char **argv) {
    int nTotalElements;  // tamanho do vetor de entrada
    int h;               // numero de faixas do histograma
    int nR;              // numero de repeticoes do kernel

    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " <nTotalElements> <h> <nR>"
                  << std::endl;
        return -1;
    }

    std::cout << "Tratando argumentos..." << std::endl;

    nTotalElements = atoi(argv[1]);
    h = atoi(argv[2]);
    // nR = atoi(argv[3]);

    std::cout << "nTotalElements: " << nTotalElements << std::endl;
    std::cout << "h: " << h << std::endl;
    // std::cout << "nR: " << nR << std::endl;

    std::cout << "\nGerando dados de entrada..." << std::endl;
    float *Input{new float[nTotalElements]};

    float nMax{-std::numeric_limits<float>::infinity()};
    float nMin{std::numeric_limits<float>::infinity()};

    for (int i = 0; i < nTotalElements; ++i) {
        int a{rand()};  // Inteiro pseudo-aleatorio entre 0 e RAND_MAX
        int b{rand()};  // Mesmo que acima

        float v{a * 100.f + b};

        if (v > nMax) {
            nMax = v;
        }
        if (v < nMin) {
            nMin = v;
        }

        // insere o valor v na posicao i do vetor de entrada
        Input[i] = v;
    }

    std::cout << "> nMin: " << nMin << std::endl;
    std::cout << "> nMax: " << nMax << std::endl;

    std::cout << "\nAlocando memoria no device..." << std::endl;

    unsigned int *OutputBH;  // matriz de saida do kernel blockHisto
    unsigned int *OutputGH;  // vetor de saida do kernel globalHisto
    unsigned int *OutputSH;  // vetor de saida da versao serial

    cudaError_t err;
    int nt;  // numero de threads por bloco
    int nb;  // numero de blocos

    // configura o numero de blocos e threads por bloco
    std::cout << "\nConfigurando numero de blocos e threads por bloco..."
              << std::endl;
    if (nTotalElements < MAX_THREADS_PER_BLOCK) {
        nt = nTotalElements;
        nb = 1;
    } else {
        nt = MAX_THREADS_PER_BLOCK;
        nb = MP * 2;
    }

    std::cout << "> nt: " << nt << std::endl;
    std::cout << "> nb: " << nb << std::endl;


    std::cout << "\nAlocando memoria no device..." << std::endl;
    // aloca matriz de saida para o kernel blockHisto (um histograma por linha)
    // - h colunas
    // - nb linhas
    // nb histogramas, cada um com h faixas (nb = NUM_BLOCOS)
    err = cudaMalloc(&OutputBH, nb * h * sizeof(unsigned int));
    if (err != cudaSuccess) {
        std::cout << "Erro alocando memoria para OutputBH: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    std::cout << "> OutputBH alocado!" << std::endl;

    // aloca vetor de saida para o kernel globalHisto (apenas um histograma)
    // - h colunas
    err = cudaMalloc(&OutputGH, h * sizeof(unsigned int));
    if (err != cudaSuccess) {
        std::cout << "Erro alocando memoria para OutputGH: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(OutputBH);
        return 1;
    }
    std::cout << "> OutputGH alocado!" << std::endl;

    // aloca vetor de saida para a versao serial (apenas um histograma)
    // - h colunas
    err = cudaMalloc(&OutputSH, h * sizeof(unsigned int));
    if (err != cudaSuccess) {
        std::cout << "Erro alocando memoria para OutputSH: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        return 1;
    }
    std::cout << "> OutputSH alocado!" << std::endl;

    // aloca vetor de entrada no device
    float *d_Input;
    err = cudaMalloc(&d_Input, nTotalElements * sizeof(float));
    if (err != cudaSuccess) {
        std::cout << "Erro alocando memoria para d_Input: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        return 1;
    }
    std::cout << "> d_Input alocado!" << std::endl;

    std::cout << "\nCopiando Input para o device..." << std::endl;
    err = cudaMemcpy(d_Input, Input, nTotalElements * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "Erro copiando dados de Input para d_Input: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        cudaFree(d_Input);
        return 2;
    }
    std::cout << "> Input copiado!" << std::endl;

    // executa o kernel blockHisto
    std::cout << "\nExecutando kernel blockHisto..." << std::endl;
    blockHisto<<<nb, nt>>>(OutputBH, h, d_Input, nTotalElements, nMin, nMax);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Erro executando blockHisto: " << cudaGetErrorString(err)
                  << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        cudaFree(d_Input);
        return 3;
    }

    // executa o kernel globalHisto
    std::cout << "\nExecutando kernel globalHisto..." << std::endl;
    globalHisto<<<nb, nt>>>(OutputGH, h, d_Input, nTotalElements, nMin, nMax);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Erro executando globalHisto: " << cudaGetErrorString(err)
                  << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        cudaFree(d_Input);
        return 3;
    }

    // executa a versao serial
    std::cout << "\nExecutando versao serial..." << std::endl;

    unsigned int *S{serialHisto(Input, nTotalElements, nMin, nMax, h)};

    std::cout << "\nCopiando dados da versao serial para OutputSH..."
              << std::endl;
    err = cudaMemcpy(OutputSH, S, h * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cout << "Erro copiando dados de S para OutputSH: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        cudaFree(d_Input);
        return 2;
    }
    std::cout << "> Dados copiados!" << std::endl;

    // valida os resultados
    std::cout << "\nValidando resultados..." << std::endl;

    // printa o input
    std::cout << "Input: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << Input[i] << " ";
    }
    std::cout << "\n\n";

    bool *result;
    cudaMallocManaged(&result, sizeof(bool));
    validaResultados<<<1, 1>>>(OutputGH, OutputBH, h, nb, OutputSH, result);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Erro executando validaResultados: "
                  << cudaGetErrorString(err) << std::endl;
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        cudaFree(d_Input);
        return 5;
    }

    // libera a memoria alocada
    cudaFree(OutputBH);
    cudaFree(OutputGH);
    cudaFree(d_Input);
    delete[] Input;

    return 0;
}
