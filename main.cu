#include <string.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "chrono.c"
#include "cuda.h"

// GTX 1080 Ti - nv00 - Constant Memory
constexpr static uint8_t MP = 28;
constexpr static uint16_t MAX_THREADS_PER_BLOCK = 1024;
constexpr static uint32_t MAX_SHARED_MEM_PER_BLOCK = 49152;
constexpr static uint32_t H_MAX = MAX_SHARED_MEM_PER_BLOCK / 4;
// Other constants
constexpr static uint8_t MAX_PRINT = 10;

__host__ __device__ constexpr float LARGURA_FAIXA(float nMin, float nMax,
                                                  int h) {
    return (nMax - nMin) / h;
}

__global__ void blockHisto(unsigned int *HH, const int h, const float *Input,
                           const uint64_t nTotalElements, const float nMin,
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
                            const uint64_t nTotalElements, const float nMin,
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

__host__ unsigned int *serialHisto(const float *Input,
                                   const uint64_t nTotalElements,
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
        std::cout << "DEBUG: H[" << i << "]: " << H[i] << "\n";
    }
#endif

    return H;
}

__global__ void validaResultados(const unsigned int *const H,
                                 const unsigned int *const HH, const int h,
                                 const int nb, const unsigned int *const S,
                                 bool *result) {
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
        printf("linha  %d: ", i + 1);
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
    uint64_t nTotalElements;  // tamanho do vetor de entrada
    int h;                    // numero de faixas do histograma
    int nR;                   // numero de repeticoes do kernel
    bool serial{true};        // se true, executa a versao serial
    chronometer_t chrono;
    double throughput;  // para medir a vazao em MFLOPS

    if (argc < 4) {
        std::cout << "Usage: " << argv[0]
                  << " <nTotalElements> <h> <nR> [--no-serial]"
                  << "\n";
        return -1;
    }

#ifdef DEBUG
    std::cout << "DEBUG: Tratando argumentos...\n";
#endif

    nTotalElements = atoi(argv[1]);
    h = atoi(argv[2]);
    nR = atoi(argv[3]);
    if (nTotalElements <= 0 || h <= 0 || nR <= 0) {
        std::cout << "Usage: " << argv[0]
                  << " <nTotalElements> <h> <nR> [--no-serial]"
                  << "\n";
        return -1;
    }

    if (h > H_MAX) {
        std::cout << "Erro: h deve ser menor ou igual a " << H_MAX << "\n";
        return -1;
    }

    if (argc == 5 && strcmp(argv[4], "--no-serial") == 0) {
        #ifdef DEBUG
        std::cout << "DEBUG: Desabilitando versão serial...\n";
        #endif
        serial = false;
    }

#ifdef DEBUG
    std::cout << "nTotalElements: " << nTotalElements << "\n";
    std::cout << "h: " << h << "\n";
    // std::cout << "nR: " << nR << "\n";

    std::cout << "\nDEBUG: Gerando dados de entrada...\n";
#endif

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

    std::cout << "> Intervalo dos valores de entrada: ";
    std::cout << "[" << nMin << ", " << nMax << "]\n";
    std::cout << "> Largura da faixa: " << LARGURA_FAIXA(nMin, nMax, h) << "\n";

#ifdef DEBUG
    std::cout << "> nMin: " << nMin << "\n";
    std::cout << "> nMax: " << nMax << "\n";

    std::cout << "\nDEBUG: Alocando memoria no device...\n";
#endif

    unsigned int *OutputBH;  // matriz de saida do kernel blockHisto
    unsigned int *OutputGH;  // vetor de saida do kernel globalHisto
    unsigned int *OutputSH;  // vetor de saida da versao serial

    cudaError_t err;
    int nt;  // numero de threads por bloco
    int nb;  // numero de blocos

#ifdef DEBUG
    std::cout << "\nDEBUG: Configurando numero de blocos e threads por bloco..."
              << "\n";
#endif
    if (nTotalElements < MAX_THREADS_PER_BLOCK) {
        nt = nTotalElements;
        nb = 1;
    } else {
        nt = MAX_THREADS_PER_BLOCK;
        nb = MP * 2;
    }

#ifdef DEBUG
    std::cout << "> nt: " << nt << "\n";
    std::cout << "> nb: " << nb << "\n";

    std::cout << "\nDEBUG: Alocando memoria no device...\n";
#endif
    // aloca matriz de saida para o kernel blockHisto (um histograma por linha)
    // - h colunas
    // - nb linhas
    // nb histogramas, cada um com h faixas (nb = NUM_BLOCOS)
    err = cudaMalloc(&OutputBH, nb * h * sizeof(unsigned int));
    if (err != cudaSuccess) {
#ifdef DEBUG
        std::cout << "DEBUG: Erro alocando memoria para OutputBH: "
                  << cudaGetErrorString(err) << "\n";
#endif
        return 1;
    }
#ifdef DEBUG
    std::cout << "> OutputBH alocado!\n";
#endif

    // aloca vetor de saida para o kernel globalHisto (apenas um histograma)
    // - h colunas
    err = cudaMalloc(&OutputGH, h * sizeof(unsigned int));
    if (err != cudaSuccess) {
#ifdef DEBUG
        std::cout << "DEBUG: Erro alocando memoria para OutputGH: "
                  << cudaGetErrorString(err) << "\n";
#endif
        cudaFree(OutputBH);
        return 1;
    }
#ifdef DEBUG
    std::cout << "> OutputGH alocado!\n";
#endif

    // aloca vetor de saida para a versao serial (apenas um histograma)
    // - h colunas
    err = cudaMalloc(&OutputSH, h * sizeof(unsigned int));
    if (err != cudaSuccess) {
#ifdef DEBUG
        std::cout << "DEBUG: Erro alocando memoria para OutputSH: "
                  << cudaGetErrorString(err) << "\n";
#endif
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        return 1;
    }
#ifdef DEBUG
    std::cout << "> OutputSH alocado!\n";
#endif

    // aloca vetor de entrada no device
    float *d_Input;
    err = cudaMalloc(&d_Input, nTotalElements * sizeof(float));
    if (err != cudaSuccess) {
#ifdef DEBUG
        std::cout << "DEBUG: Erro alocando memoria para d_Input: "
                  << cudaGetErrorString(err) << "\n";
#endif
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        return 1;
    }
#ifdef DEBUG
    std::cout << "> d_Input alocado!\n";
#endif

#ifdef DEBUG
    std::cout << "\nDEBUG: Copiando Input para o device...\n";
#endif
    err = cudaMemcpy(d_Input, Input, nTotalElements * sizeof(float),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
#ifdef DEBUG
        std::cout << "DEBUG: Erro copiando dados de Input para d_Input: "
                  << cudaGetErrorString(err) << "\n";
#endif
        cudaFree(OutputBH);
        cudaFree(OutputGH);
        cudaFree(OutputSH);
        cudaFree(d_Input);
        return 2;
    }
#ifdef DEBUG
    std::cout << "> Input copiado!\n";
#endif

    chrono_reset(&chrono);
    chrono_start(&chrono);

// executa o kernel blockHisto
#ifdef DEBUG
    std::cout << "\nDEBUG: Executando kernel blockHisto...\n";
#endif
    for (int i{0}; i < nR; ++i) {
        cudaMemset(OutputBH, 0, nb * h * sizeof(unsigned int));
        blockHisto<<<nb, nt>>>(OutputBH, h, d_Input, nTotalElements, nMin,
                               nMax);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
#ifdef DEBUG
            std::cout << "DEBUG: Erro executando blockHisto: "
                      << cudaGetErrorString(err) << "\n";
#endif
            cudaFree(OutputBH);
            cudaFree(OutputGH);
            cudaFree(OutputSH);
            cudaFree(d_Input);
            return 3;
        }
    }

    cudaDeviceSynchronize();
    chrono_stop(&chrono);
    chrono_report_TimeInLoop(&chrono, "BlockHisto", nR);
    // ==> each op takes 15085 ns
    throughput = (nTotalElements * nR) / (chrono.xtotal_ns / 1e3);
    std::cout << "        ==> throughput: " << throughput << " MFLOP/s\n";

    chrono_reset(&chrono);
    chrono_start(&chrono);

// executa o kernel globalHisto
#ifdef DEBUG
    std::cout << "\nDEBUG: Executando kernel globalHisto...\n";
#endif
    for (int i{0}; i < nR; ++i) {
        cudaMemset(OutputGH, 0, h * sizeof(unsigned int));
        globalHisto<<<nb, nt>>>(OutputGH, h, d_Input, nTotalElements, nMin,
                                nMax);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
#ifdef DEBUG
            std::cout << "DEBUG: Erro executando globalHisto: "
                      << cudaGetErrorString(err) << "\n";
#endif
            cudaFree(OutputBH);
            cudaFree(OutputGH);
            cudaFree(OutputSH);
            cudaFree(d_Input);
            return 3;
        }
    }

    cudaDeviceSynchronize();
    chrono_stop(&chrono);
    chrono_report_TimeInLoop(&chrono, "GlobalHisto", nR);
    throughput = (nTotalElements * nR) / (chrono.xtotal_ns / 1e3);
    std::cout << "        ==> throughput: " << throughput << " MFLOP/s\n";

    unsigned int *S;

    chrono_reset(&chrono);
    chrono_start(&chrono);

    // executa a versão serial
    if (serial) {
#ifdef DEBUG
        std::cout << "\nDEBUG: Executando versão serial...\n";
#endif
        for (int i{0}; i < nR; ++i) {
            S = serialHisto(Input, nTotalElements, nMin, nMax, h);
            if (i != nR - 1) delete[] S;
        }

        chrono_stop(&chrono);
        chrono_report_TimeInLoop(&chrono, "SerialHisto", nR);
        throughput = (nTotalElements * nR) / (chrono.xtotal_ns / 1e3);
        std::cout << "        ==> throughput: " << throughput << " MFLOP/s\n";

#ifdef DEBUG
        std::cout << "\nDEBUG: Copiando dados da versão serial para OutputSH..."
                  << "\n";
#endif
        err = cudaMemcpy(OutputSH, S, h * sizeof(unsigned int),
                         cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
#ifdef DEBUG
            std::cout << "DEBUG: Erro copiando dados de S para OutputSH: "
                      << cudaGetErrorString(err) << "\n";
#endif
            cudaFree(OutputBH);
            cudaFree(OutputGH);
            cudaFree(OutputSH);
            cudaFree(d_Input);
            return 2;
        }
#ifdef DEBUG
        std::cout << "> Dados copiados!\n";
#endif

// valida os resultados
#ifdef VALIDAR
        std::cout << "\nDEBUG: Validando resultados...\n";

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
#ifdef DEBUG
            std::cout << "DEBUG: Erro executando validaResultados: "
                      << cudaGetErrorString(err) << "\n";
#endif
            cudaFree(OutputBH);
            cudaFree(OutputGH);
            cudaFree(OutputSH);
            cudaFree(d_Input);
            return 5;
        }
#endif  // VALIDAR
    }

    // libera a memoria alocada
    cudaFree(OutputBH);
    cudaFree(OutputGH);
    cudaFree(OutputSH);
    cudaFree(d_Input);
    delete[] Input;

    return 0;
}
