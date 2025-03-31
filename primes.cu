/*
 * nvcc -O3 -std=c++17 -arch=compute_80 -Invidia-mathdx-25.01.1/nvidia/mathdx/25.01/include primes.cu -o primes_program && ./primes_program 
 */

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>
#include <tuple>
#include <stdio.h>
#include <cassert>
#include <bitset>
#include <filesystem>
#include <thread>
#include <mutex>
#include <atomic>
#include <ciso646> // fix missing `not` for `fft_execution.hpp`
#define CUFFTDX_DISABLE_CUTLASS_DEPENDENCY
#include <cufftdx.hpp>

#include <cuda/std/limits>
#include <thrust/binary_search.h>
#include <thrust/device_ptr.h>

using namespace cufftdx;

#define WARP_SIZE 32
#define PRIME_LOWER_BOUND (1 << 29)
#define VALID_PAIR_COUNT_BUFFER_SIZE (1 << 25)
#define PAIR_SCORE_BUFFER_SIZE (1 << 30)
#define MAX_SCORED_PRIME_COUNT (1 << 19)
#define EXECUTE_TEST_RUN false

#define BIT_RANGE 31
#define HASH_RANGE_1D 32
#define HASH_RANGE_2D 32

#define TARGET_PRIME_GROUP_SIZE 4
#define COMBINATIONS_TO_OUTPUT 100

using FFT_B = decltype(Thread() + Size<BIT_RANGE>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>());
using FFT_XY = decltype(Thread() + Size<HASH_RANGE_2D * 2>() + Precision<float>() + Type<fft_type::c2c>() + Direction<fft_direction::forward>());

using uint = unsigned int;

template<typename T>
struct pair {
    T a, b;
};

template<typename TKey, typename TValue>
struct heapEntry {
public:
    TKey key;
    TValue value;
};
using primePairHeapEntry = heapEntry<float, pair<int>>;

bool fileExists(const std::string& filename) {
    return std::filesystem::exists(filename);
}

template<int InterleaveStride>
__host__ __device__ int indexInterleavedHeap(int iHeapOneIndexed) {
    return (iHeapOneIndexed - 1) * InterleaveStride;
}

template<typename TKey, typename TValue, int HeapSize, int InterleaveStride>
__host__ __device__ void heapReplaceSiftDown(TKey key, TValue value, heapEntry<TKey, TValue>* heapStart) {
    if (!(key < heapStart[0].key)) return;
    
    heapEntry<TKey, TValue> top = heapEntry<TKey, TValue> { key, value };
    int indexTop = 0;
    heapStart[indexTop] = top;
    
    int iHeap = 1;
    while (iHeap * 2 <= HeapSize) {
        int indexLeft = indexInterleavedHeap<InterleaveStride>(iHeap * 2);
        int indexRight = indexInterleavedHeap<InterleaveStride>(iHeap * 2 + 1);
        heapEntry<TKey, TValue> left = heapStart[indexLeft];
        heapEntry<TKey, TValue> right = (iHeap * 2 + 1 <= HeapSize) ?
            heapStart[indexRight] :
            heapEntry<TKey, TValue> { -INFINITY, TValue { } }; // Note: this will break if TKey is not a float type, but we're only using floats rn.
        bool useRight = (right.key > left.key);
        heapEntry<TKey, TValue> chosen = useRight ? right : left;
        int indexChosen = useRight ? indexRight : indexLeft;
        if (!(chosen.key > top.key)) break;
        
        heapStart[indexTop] = chosen;
        heapStart[indexChosen] = top;
        iHeap = iHeap * 2 + (useRight ? 1 : 0);
        indexTop = indexChosen;
    }
}

__device__ uint64_t sqrt(uint64_t x) {
    if (x == 0 || x == 1) return x;

    uint64_t left = 1LL << ((63 - __clzll(x)) / 2);
    uint64_t right = left << 1;
    uint64_t result = 0;

    while (left <= right) {
        uint64_t mid = left + (right - left) / 2;
        uint64_t midSquared = mid * mid;

        if (midSquared == x) return mid;
        else if (midSquared < x) {
            left = mid + 1;
            result = mid;
        } else right = mid - 1;
    }
    
    return result;
}

__host__ __device__ pair<int> extractIndexPair(int64_t pairIndex) {
    int indexA = ((int)sqrt((double)pairIndex * 8.0 + 1.0) - 1) / 2 + 1;
    int indexB = (int)pairIndex - indexA * (indexA - 1) / 2;
    return pair<int> { indexA, indexB };
}

__device__ pair<int> extractIndexPair_fullPrecision(int64_t pairIndex) {
    int indexA = ((int)sqrt((uint64_t)pairIndex * 8 + 1) - 1) / 2 + 1;
    int indexB = (int)(pairIndex - (int64_t)indexA * (indexA - 1) / 2);
    return pair<int> { indexA, indexB };
}

inline int64_t mergeIndexPair(int indexA, int indexB) {
    return (int64_t)indexA * (indexA - 1) / 2 + indexB;
}

// Computes the average normalized squared error (+1, /N) from the average in a single loop
// by using rearranged expanded terms. score = [[Σ (Value/Average-1)²] / N + 1] / N
// The +1 is to remove a -1 from the final formula, improving numerical stability,
// with no sacrifice to the mathematical correctness of relative comparisons.
// The external /N similarly removes the need to consider the total count.
struct fourierSpectrumScorer {
    double spectralValueSum = 0;
    double spectralValueSumOfSquares = 0;
    
    __host__ __device__ void accept(double real, double imag) {
        double spectralValue = real * real + imag * imag;
        spectralValueSum += spectralValue;
        spectralValueSumOfSquares += spectralValue * spectralValue;
    }
    
    __host__ __device__ double compute() {
        return spectralValueSumOfSquares / (spectralValueSum * spectralValueSum);
    }
};

// Enhanced Fourier analyzer optimized for coherent noise generation
// This analyzer evaluates additional properties important for natural-looking noise:
// 1. Spectral falloff (natural noise has 1/f falloff)
// 2. Anisotropy detection (directional bias - coherent noise should be isotropic)
// 3. Phase coherence (random phases are better for noise)
// 4. Gradient uniformity (for proper gradient distribution)
struct enhancedFourierAnalyzer {
    // Statistical accumulation variables (from original analyzer)
    double spectralValueSum = 0;
    double spectralValueSumOfSquares = 0;
    
    // For spectral falloff (natural noise has 1/f falloff)
    double lowFreqEnergySum = 0;  // Lower 1/3 of spectrum
    double midFreqEnergySum = 0;  // Middle 1/3 of spectrum
    double highFreqEnergySum = 0; // Upper 1/3 of spectrum
    
    // For anisotropy detection (directional bias)
    double horizontalEnergySum = 0;
    double verticalEnergySum = 0;
    double diagonalEnergySum = 0;
    
    // For phase coherence analysis
    double cosPhaseSum = 0;
    double sinPhaseSum = 0;
    
    // Frequency bin counters
    int lowFreqCount = 0;
    int midFreqCount = 0;
    int highFreqCount = 0;
    int horizontalCount = 0;
    int verticalCount = 0;
    int diagonalCount = 0;
    int totalCount = 0;
    
    __host__ __device__ void accept(double real, double imag, int xFreq, int yFreq, int maxFreq) {
        double spectralValue = real * real + imag * imag;
        totalCount++;
        
        // Original metrics
        spectralValueSum += spectralValue;
        spectralValueSumOfSquares += spectralValue * spectralValue;
        
        // Calculate frequency magnitude
        double freqMagnitude = sqrt((double)(xFreq*xFreq + yFreq*yFreq));
        double freqNormalized = freqMagnitude / maxFreq;
        
        // Spectral falloff analysis (divide spectrum into 3 bands)
        if (freqNormalized < 0.33) {
            lowFreqEnergySum += spectralValue;
            lowFreqCount++;
        } else if (freqNormalized < 0.67) {
            midFreqEnergySum += spectralValue;
            midFreqCount++;
        } else {
            highFreqEnergySum += spectralValue;
            highFreqCount++;
        }
        
        // Anisotropy analysis (directional bias)
        if (abs(xFreq) > 3 * abs(yFreq)) {
            horizontalEnergySum += spectralValue;
            horizontalCount++;
        } else if (abs(yFreq) > 3 * abs(xFreq)) {
            verticalEnergySum += spectralValue;
            verticalCount++;
        } else if (abs(xFreq) > 0.5 * abs(yFreq) && abs(yFreq) > 0.5 * abs(xFreq)) {
            diagonalEnergySum += spectralValue;
            diagonalCount++;
        }
        
        // Phase coherence
        if (spectralValue > 0) {
            double phase = atan2(imag, real);
            cosPhaseSum += cos(phase);
            sinPhaseSum += sin(phase);
        }
    }
    
    // Simple accept method that works with the original code (no frequency information)
    __host__ __device__ void accept(double real, double imag) {
        double spectralValue = real * real + imag * imag;
        
        // Original metrics
        spectralValueSum += spectralValue;
        spectralValueSumOfSquares += spectralValue * spectralValue;
        totalCount++;
    }
    
    __host__ __device__ double computeBasicScore() {
        // Original uniformity score (lower is better)
        return spectralValueSumOfSquares / (spectralValueSum * spectralValueSum);
    }
    
    __host__ __device__ double computeSpectralFalloff() {
        // Ideal coherent noise has energy falloff from low to high frequencies
        // We want low > mid > high energy density (adjusted for bin count)
        double lowFreqDensity = lowFreqCount > 0 ? lowFreqEnergySum / lowFreqCount : 0;
        double midFreqDensity = midFreqCount > 0 ? midFreqEnergySum / midFreqCount : 0;
        double highFreqDensity = highFreqCount > 0 ? highFreqEnergySum / highFreqCount : 0;
        
        // Calculate falloff ratios (ideally around 2:1 between adjacent bands)
        double lowMidRatio = midFreqDensity > 0 ? lowFreqDensity / midFreqDensity : 0;
        double midHighRatio = highFreqDensity > 0 ? midFreqDensity / highFreqDensity : 0;
        
        // Penalize deviation from ideal 2:1 ratios
        double falloffScore = fabs(lowMidRatio - 2.0) + fabs(midHighRatio - 2.0);
        return falloffScore;
    }
    
    __host__ __device__ double computeAnisotropyScore() {
        // Calculate energy density in each direction
        double horizDensity = horizontalCount > 0 ? horizontalEnergySum / horizontalCount : 0;
        double vertDensity = verticalCount > 0 ? verticalEnergySum / verticalCount : 0;
        double diagDensity = diagonalCount > 0 ? diagonalEnergySum / diagonalCount : 0;
        
        // Calculate variance between directions (lower is better - means more isotropic)
        double meanDensity = (horizDensity + vertDensity + diagDensity) / 3.0;
        double varianceDensity = 0;
        
        if (meanDensity > 0) {
            double horizDev = horizDensity / meanDensity - 1.0;
            double vertDev = vertDensity / meanDensity - 1.0;
            double diagDev = diagDensity / meanDensity - 1.0;
            
            varianceDensity = (horizDev*horizDev + vertDev*vertDev + diagDev*diagDev) / 3.0;
        }
        
        return varianceDensity;
    }
    
    __host__ __device__ double computePhaseCoherenceScore() {
        // Measure randomness of phases (higher is better for noise)
        // This returns lower values for random phases (good for noise)
        if (totalCount <= 1) return 0.0;
        
        double phaseCoherence = sqrt(cosPhaseSum*cosPhaseSum + sinPhaseSum*sinPhaseSum) / totalCount;
        return phaseCoherence;
    }
    
    __host__ __device__ double compute() {
        // If we have frequency information, use the detailed metrics
        if (lowFreqCount > 0 || horizontalCount > 0) {
            double uniformityScore = computeBasicScore();
            double falloffScore = computeSpectralFalloff();
            double anisotropyScore = computeAnisotropyScore();
            double phaseCoherenceScore = computePhaseCoherenceScore();
            
            // Combine scores with appropriate weights for coherent noise
            // - We want uniform energy within each frequency band (low uniformityScore)
            // - We want natural falloff between bands (low falloffScore)
            // - We want isotropic properties (low anisotropyScore)
            // - We want random phases (low phaseCoherenceScore)
            
            return uniformityScore + 
                   0.5 * falloffScore + 
                   2.0 * anisotropyScore + 
                   3.0 * phaseCoherenceScore;
        } else {
            // Fall back to basic score if we don't have frequency information
            return computeBasicScore();
        }
    }
};


// Enhanced gradient analyzer focused on index distribution for table lookups
struct indexDistributionAnalyzer {
    // Analysis for index distribution (e.g., for 16-entry lookup tables)
    static const int INDEX_BINS = 16; // Assuming low 4 bits of hash used for lookup index
    double indexBins[INDEX_BINS];
    int totalHashes;

    __host__ __device__ indexDistributionAnalyzer() {
        for (int i = 0; i < INDEX_BINS; i++) {
            indexBins[i] = 0;
        }
        totalHashes = 0;
    }

    __host__ __device__ void analyzeHash(int hash) {
        // Extract the likely index bits from the hash
        // Assuming the low 4 bits determine the index for a 16-entry lookup
        int index = hash & (INDEX_BINS - 1); // Equivalent to hash % 16

        // Increment the count for the corresponding bin
        if (index >= 0 && index < INDEX_BINS) { // Basic bounds check
           indexBins[index]++;
        }
        totalHashes++;
    }

    __host__ __device__ double computeUniformity() {
        // Compute uniformity based on how evenly the indices are distributed
        if (totalHashes == 0) return 0.0;

        double idealCount = (double)totalHashes / INDEX_BINS;
        double sumSquaredDev = 0;

        for (int i = 0; i < INDEX_BINS; i++) {
            double deviation = indexBins[i] - idealCount;
            sumSquaredDev += deviation * deviation;
        }

        // Return a Chi-squared-like metric. Lower is better (more uniform).
        // Avoid division by zero if idealCount is somehow zero.
        return (idealCount > 1e-9) ? (sumSquaredDev / idealCount) : 0.0;
    }
};

/*
 * Prime sieving
 */

std::vector<uint> getPrimes() {
    const std::string filename = "primes.bin";
    std::vector<uint> primes;
    
    if (!fileExists(filename)) {
        std::cout << "File `" << filename << "` does not exist. Generating list of primes..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        
        int givenPrimes[] { 2, 3, 5, 7, 11, 13, 17, 19 };
        int smallPrimeMultipleSectionSize = 1;
        for (int i = 0; i < sizeof(givenPrimes) / sizeof(int); i++) {
            smallPrimeMultipleSectionSize *= givenPrimes[i];
            primes.push_back(givenPrimes[i]);
        }
        
        int smallPrimeNonMultipleCount = smallPrimeMultipleSectionSize - 1;
        std::vector<int> smallPrimeNonMultipleIndexMap;
        std::vector<int> smallPrimeNonMultiples;
        {
            std::vector<bool> smallPrimeMultipleSieve(smallPrimeMultipleSectionSize, false);
            smallPrimeMultipleSieve[0] = true;
            for (int givenPrimeIndex = 0; givenPrimeIndex < sizeof(givenPrimes) / sizeof(int); givenPrimeIndex++) {
                int givenPrime = givenPrimes[givenPrimeIndex];
                for (int multiple = givenPrime; multiple < smallPrimeMultipleSectionSize; multiple += givenPrime) {
                    if (!smallPrimeMultipleSieve[multiple]) {
                        smallPrimeMultipleSieve[multiple] = true;
                        smallPrimeNonMultipleCount--;
                    }
                }
            }
            
            smallPrimeNonMultipleIndexMap = std::vector<int>(smallPrimeMultipleSectionSize, -1);
            smallPrimeNonMultiples = std::vector<int>(smallPrimeNonMultipleCount);
            for (int iSieve = 0, iNonMultiple = 0; iSieve < smallPrimeMultipleSectionSize; iSieve++) {
                if (smallPrimeMultipleSieve[iSieve]) continue;
                
                smallPrimeNonMultipleIndexMap[iSieve] = iNonMultiple;
                smallPrimeNonMultiples[iNonMultiple] = iSieve;
                iNonMultiple++;
            }
        }
        
        int64_t targetTotalSieveSize = 1LL << 32;
        int compressedTotalSieveSectionCount = (int)((targetTotalSieveSize + smallPrimeMultipleSectionSize - 1) / smallPrimeMultipleSectionSize);
        int compressedTotalSieveSize = compressedTotalSieveSectionCount * smallPrimeNonMultipleCount;
        std::vector<bool> compressedTotalSieve(compressedTotalSieveSize, false);
        compressedTotalSieve[0] = true;
        
        for (int sectionIndex = 0, indexInSection = 1; sectionIndex < compressedTotalSieveSectionCount; sectionIndex++, indexInSection = 0) {
            for (; indexInSection < smallPrimeNonMultipleCount; indexInSection++) {
                int compressedTotalSieveIndex = sectionIndex * smallPrimeNonMultipleCount + indexInSection;
                
                int64_t potentialPrime = sectionIndex * (int64_t)smallPrimeMultipleSectionSize + smallPrimeNonMultiples[indexInSection];
                if (potentialPrime >= targetTotalSieveSize) goto ExitLoops;
                if (compressedTotalSieve[compressedTotalSieveIndex]) continue;
                
                primes.push_back((uint)potentialPrime);
                for (int64_t knownComposite = potentialPrime * 2; knownComposite < targetTotalSieveSize; knownComposite += potentialPrime) {
                    int knownCompositeIndexOffset = smallPrimeNonMultipleIndexMap[knownComposite % smallPrimeMultipleSectionSize];
                    if (knownCompositeIndexOffset < 0) continue;
                    
                    int knownCompositeCompressedTotalSieveIndex = (int)(knownComposite / smallPrimeMultipleSectionSize) * smallPrimeNonMultipleCount + knownCompositeIndexOffset;
                    compressedTotalSieve[knownCompositeCompressedTotalSieveIndex] = true;
                }
            }
        }
        ExitLoops:
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Sieving took: " << duration.count() << " second" << (duration.count() == 1 ? "" : "s") << std::endl;
        std::cout << "Total primes found: " << primes.size() << std::endl;
        
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(primes.data()), primes.size() * sizeof(int));
            outFile.close();
            std::cout << "Wrote primes to file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for writing: " << filename << std::endl;
        }
    } else {
        std::ifstream inFile(filename, std::ios::binary);
        if (inFile.is_open()) {
            inFile.seekg(0, std::ios::end);
            size_t fileSize = inFile.tellg();
            inFile.seekg(0, std::ios::beg);
            
            size_t numInts = fileSize / sizeof(int);
            primes.resize(numInts);
            inFile.read(reinterpret_cast<char*>(primes.data()), fileSize);
            inFile.close();
            std::cout << "Loaded " << primes.size() << " prime" << (primes.size() == 1 ? "" : "s") << " from file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for reading: " << filename << std::endl;
        }
    }
    
    return primes;
}

/*
 * Prime narrowing
 */

struct narrowPrimes_sharedMemoryPointers {
    uint* warpActiveLoopThreadMasksPtr;
    uint8_t* endPtr;
    
    __host__ __device__ static narrowPrimes_sharedMemoryPointers allocate(uint8_t* sharedPtr, int warpsPerBlock) {
        uint8_t* incrementingSharedPtr = sharedPtr;
        auto allocate = [&incrementingSharedPtr](size_t size) {
            uint8_t* allocatedPtr = incrementingSharedPtr;
            incrementingSharedPtr += size;
            return allocatedPtr;
        };
        narrowPrimes_sharedMemoryPointers value;
        value.warpActiveLoopThreadMasksPtr = (uint*)allocate(sizeof(int) * warpsPerBlock);
        value.endPtr = incrementingSharedPtr;
        return value;
    }
};

__global__ void narrowPrimesKernel(int* d_primes, int primeCount, size_t batchSize, int* d_lock, heapEntry<float, uint>* d_primeEntryHeap) {
	using complex_type = FFT_B::value_type;
    
    extern __shared__ uint8_t sharedMemoryStart[];
    narrowPrimes_sharedMemoryPointers shared = narrowPrimes_sharedMemoryPointers::allocate(&sharedMemoryStart[0], blockDim.y);

    if (threadIdx.x == 0) {
        shared.warpActiveLoopThreadMasksPtr[threadIdx.y] = 0xFFFFFFFF;
    }
    int iThread = threadIdx.y * blockDim.x + threadIdx.x;
    for (int i = iThread; i < MAX_SCORED_PRIME_COUNT; i += blockDim.x * blockDim.y) {
        d_primeEntryHeap[i] = heapEntry<float, uint>{ INFINITY, 0 }; // TODO move this to the host; can't guarantee
    }
    
    __syncthreads();

    int iBatch = blockIdx.x * (blockDim.x * blockDim.y) + iThread;
    size_t startIndex = iBatch * batchSize;
	
	complex_type buffer[BIT_RANGE * HASH_RANGE_1D];
	complex_type buffer_local_double[HASH_RANGE_1D * 2];
    
	size_t iWithinBatch = 0;
    for (;;) {
        bool loopCondition = (iWithinBatch < batchSize);
        if (!loopCondition) {
            atomicAnd(&shared.warpActiveLoopThreadMasksPtr[threadIdx.y], ~(1 << threadIdx.x));
        }
        __syncwarp(shared.warpActiveLoopThreadMasksPtr[threadIdx.y]); // Make sure no update syncs are reached before all warp threads agree on the above.
        if (!loopCondition) break;
        
		int prime = d_primes[startIndex + iWithinBatch];
		
		int xorBase = prime * -(HASH_RANGE_1D / 2);
		for (int xBufferIndex = 0; xBufferIndex < HASH_RANGE_1D; xBufferIndex++) {
            int bufferIndexWithX = xBufferIndex * BIT_RANGE;
            int hash = xorBase ^= (xorBase >> 15);
            int hashShifted = hash;
            for (int bitIndex = 0; bitIndex < BIT_RANGE; bitIndex++) {
                int bufferIndexWithXB = bitIndex + bufferIndexWithX;
                float value = (hashShifted & 1) ? -1.0f : 1.0f;
                buffer[bufferIndexWithXB] = complex_type{ value, 0.0f };
                hashShifted >>= 1;
            }
            xorBase += prime;
            
            // Compute the Discrete Fourier Transform over the bit string we just wrote at this X,Y position.
            FFT_B().execute(&buffer[bufferIndexWithX]);
		}
		
		// Compute the Discrete Fourier Transform over the X direction for all Y and bit positions.
        enhancedFourierAnalyzer analyzer;
        indexDistributionAnalyzer gradientAnalyzer;
        
        for (int bitIndex = 0; bitIndex < BIT_RANGE; bitIndex++) {
            int bufferIndexWithB = bitIndex;
            for (int xBufferIndex = 0; xBufferIndex < HASH_RANGE_1D; xBufferIndex++) buffer_local_double[xBufferIndex] = buffer[bufferIndexWithB + xBufferIndex * BIT_RANGE];
            for (int xBufferIndex = HASH_RANGE_1D; xBufferIndex < HASH_RANGE_1D * 2; xBufferIndex++) buffer_local_double[xBufferIndex] = complex_type{ 0.0, 0.0 };
            FFT_XY().execute(&buffer_local_double[0]);
            
            for (int xBufferIndex = 0; xBufferIndex < HASH_RANGE_1D; xBufferIndex++) {
                complex_type& value = buffer_local_double[xBufferIndex];
                
                // Convert buffer indices to frequency values
                int xFreq = xBufferIndex;
                if (xFreq >= HASH_RANGE_1D / 2) xFreq -= HASH_RANGE_1D;
                
                // Since this is 1D analysis, y frequency is 0
                analyzer.accept(value.x, value.y, xFreq, 0, HASH_RANGE_1D/2);
            }
        }
        
        // Analyze gradient distribution for this prime
        for (int i = 0; i < 64; i++) {
            int hash = prime * (i + 1);
            hash ^= (hash >> 15);
            gradientAnalyzer.analyzeHash(hash);
        }
		
        // Compute combined score with components for coherent noise quality
        double spectralScore = analyzer.compute();
        double gradientScore = gradientAnalyzer.computeUniformity();
        
        // Combine scores with weights adjusted for coherent noise quality
        double score = 0.8 * spectralScore + 0.2 * gradientScore;
        
        // Update heap, one thread at a time
        // Note: heap too large for per-warp heap approach.
        {
            int chosenWarpThread = __ffs(shared.warpActiveLoopThreadMasksPtr[threadIdx.y]) - 1;
            if (chosenWarpThread == threadIdx.x) {
                while (atomicCAS(d_lock, 0, 1) != 0);
            }
            __syncwarp(shared.warpActiveLoopThreadMasksPtr[threadIdx.y]);
            
            // Chosen in place of per-thread locks, to avoid inter-warp deadlock.
            for (int i = 0; i < WARP_SIZE; i++) {
                if (i == threadIdx.x) {
                    heapReplaceSiftDown<float, uint, MAX_SCORED_PRIME_COUNT, 1>((float)score, (uint)prime, &d_primeEntryHeap[0]);
                }
                __syncwarp(shared.warpActiveLoopThreadMasksPtr[threadIdx.y]);
            }
            
            if (chosenWarpThread == threadIdx.x) {
                *d_lock = 0;
            }
        }
        
        if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && ((batchSize - iWithinBatch) & 0x7F) == 1) {
            printf("[Thread 0] Progress: %llu / %llu (%.4f%)\n", iWithinBatch + 1, batchSize, (iWithinBatch + 1) * 100.0 / batchSize);
        }
		
        iWithinBatch++;
	}
}

std::vector<uint> narrowPrimes(cudaDeviceProp& deviceProp, std::vector<uint>& primes) {
    
    const std::string filename = "primesNarrowed_" + std::to_string(MAX_SCORED_PRIME_COUNT) + "_" + std::to_string(PRIME_LOWER_BOUND) + "_" + std::to_string(HASH_RANGE_1D) + ".bin";
    std::vector<uint> primesNarrowed;
    primesNarrowed.reserve(MAX_SCORED_PRIME_COUNT);
    
    if (!fileExists(filename)) {
    
        auto primeLowerBoundIndex = std::lower_bound(primes.begin(), primes.end(), (uint)PRIME_LOWER_BOUND);
        int primeStartConsidered = primeLowerBoundIndex - primes.begin();
        int primeCountConsidered = primes.end() - primeLowerBoundIndex;
        
        std::cout << "Prime count: " << primes.size() << std::endl;
        std::cout << "Prime lower bound of 0x" << std::hex << std::uppercase << PRIME_LOWER_BOUND << " applied; consideration size truncated to: " << std::dec << primeCountConsidered << " (" << (primeCountConsidered * 100.0 / primes.size()) << "%)" << std::endl;
        std::cout << "First prime: " << primes[primeStartConsidered] << " (0x" << std::hex << std::uppercase << primes[primeStartConsidered] << ")" << std::dec << std::endl;

        int blockCount = 0;
        int threadsPerBlock = 0;
        {
            auto status = cudaOccupancyMaxPotentialBlockSize(&blockCount, &threadsPerBlock, narrowPrimesKernel, 0, 0);
            if (status != cudaSuccess) {
                std::cout << "CUDA block size query error: " << cudaGetErrorString(status) << std::endl;
            }
        }

        int warpsPerBlock = threadsPerBlock / WARP_SIZE;
        size_t sharedMemorySizePerBlock = (size_t)narrowPrimes_sharedMemoryPointers::allocate((uint8_t*)0, warpsPerBlock).endPtr;
        
        int threadCount = threadsPerBlock * blockCount;
        size_t batchSize = (primeCountConsidered + threadCount - 1) / threadCount;
        size_t totalCoverage = batchSize * threadCount;
        
        std::cout << "CUDA grid size in blocks: " << blockCount << std::endl;
        std::cout << "CUDA block size in threads: " << threadsPerBlock << std::endl;
        std::cout << "CUDA warp size in threads: " << deviceProp.warpSize << " (expected: " << WARP_SIZE << ")" << std::endl;
        std::cout << "CUDA warps per block: " << warpsPerBlock << std::endl;
        std::cout << "Shared memory size per block: " << sharedMemorySizePerBlock << std::endl;
        std::cout << "CUDA thread count: " << blockCount << " * " << threadsPerBlock << " = " << threadCount << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Total index coverage: " << totalCoverage << std::endl;
        
        uint* d_primes;
        cudaMalloc((void**)&d_primes, primeCountConsidered * sizeof(uint));
        cudaMemcpy(d_primes, primes.data() + primeStartConsidered, primeCountConsidered * sizeof(uint), cudaMemcpyHostToDevice);
        
        int* d_lock;
        cudaMalloc((void**)&d_lock, sizeof(int));
        cudaMemset(d_lock, 0, sizeof(int));
        
        std::vector<heapEntry<float, uint>> primeHeap;
        heapEntry<float, uint>* d_primeHeap;
        cudaMalloc((void**)&d_primeHeap, MAX_SCORED_PRIME_COUNT * sizeof(heapEntry<float, uint>));
        
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        std::cout << "Running prime narrowing kernel..." << std::endl;
        narrowPrimesKernel<<<blockCount, dim3(WARP_SIZE, warpsPerBlock), sharedMemorySizePerBlock>>>(
            (int*)d_primes, primeCountConsidered, batchSize, d_lock, d_primeHeap);
        
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("Completed. Kernel elapsed time: %.3f s \n", time / 1000);
        
        cudaDeviceSynchronize();
        {
            auto status = cudaGetLastError();
            if (status != cudaSuccess) {
                std::cout << "CUDA kernel execution error: " << cudaGetErrorString(status) << std::endl;
                exit(-1);
            }
        }
        
        primeHeap.resize(MAX_SCORED_PRIME_COUNT);
        cudaMemcpy(primeHeap.data(), d_primeHeap, MAX_SCORED_PRIME_COUNT * sizeof(pair<uint>), cudaMemcpyDeviceToHost);
        
        std::transform(primeHeap.begin(), primeHeap.end(), std::back_inserter(primesNarrowed), [](const heapEntry<float, uint>& entry) { return entry.value; });
        std::sort(primesNarrowed.begin(), primesNarrowed.end());
        
        cudaFree(d_primeHeap);
        cudaFree(d_lock);
        cudaFree(d_primes);
        
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(primesNarrowed.data()), primesNarrowed.size() * sizeof(int));
            outFile.close();
            std::cout << "Wrote narrowed primes to file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for writing: " << filename << std::endl;
        }
    } else {
        std::ifstream inFile(filename, std::ios::binary);
        if (inFile.is_open()) {
            inFile.seekg(0, std::ios::end);
            size_t fileSize = inFile.tellg();
            inFile.seekg(0, std::ios::beg);
            
            size_t numInts = fileSize / sizeof(int);
            primesNarrowed.resize(numInts);
            inFile.read(reinterpret_cast<char*>(primesNarrowed.data()), fileSize);
            inFile.close();
            std::cout << "Loaded " << primesNarrowed.size() << " prime" << (primesNarrowed.size() == 1 ? "" : "s") << " from file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for reading: " << filename << std::endl;
        }
    }
    
    return primesNarrowed;
}

/*
 * Prime/Multiplier discovery
 */

struct findValidPrimeMultiplierPairs_sharedMemoryPointers {
    int* lockPtr;
    uint* warpActiveLoopThreadMasksPtr;
    uint8_t* endPtr;
    
    __host__ __device__ static findValidPrimeMultiplierPairs_sharedMemoryPointers allocate(uint8_t* sharedPtr, int warpsPerBlock) {
        uint8_t* incrementingSharedPtr = sharedPtr;
        auto allocate = [&incrementingSharedPtr](size_t size) {
            uint8_t* allocatedPtr = incrementingSharedPtr;
            incrementingSharedPtr += size;
            return allocatedPtr;
        };
        findValidPrimeMultiplierPairs_sharedMemoryPointers value;
        value.lockPtr = (int*)allocate(sizeof(int));
        value.warpActiveLoopThreadMasksPtr = (uint*)allocate(sizeof(int) * warpsPerBlock);
        value.endPtr = incrementingSharedPtr;
        return value;
    }
};

struct findValidPrimeMultiplierPairs_vars {
    int validPairCount;
    bool stoppedShort;
};

__global__ void findValidPrimeMultiplierPairs(
    uint* d_primes, int primeCount, int64_t batchSize,
    int64_t* d_threadIndicesWithinBatch, findValidPrimeMultiplierPairs_vars* d_vars, pair<uint>* d_validPairs) {
    
    extern __shared__ uint8_t sharedMemoryStart[];
    findValidPrimeMultiplierPairs_sharedMemoryPointers shared = findValidPrimeMultiplierPairs_sharedMemoryPointers::allocate(&sharedMemoryStart[0], blockDim.y);
    
    __syncthreads();

    int iThreadInBlock = threadIdx.y * blockDim.x + threadIdx.x;
    int iBatch = blockIdx.x * (blockDim.x * blockDim.y) + iThreadInBlock;
    auto startIndices = extractIndexPair_fullPrecision((int64_t)iBatch * batchSize);
    
    int64_t iWithinBatch = d_threadIndicesWithinBatch[iBatch];
    for (int indexA = startIndices.a, indexB = startIndices.b;;) {
        bool loopCondition = (iWithinBatch < batchSize && indexA < primeCount);
        if (!loopCondition) {
            atomicAnd(&shared.warpActiveLoopThreadMasksPtr[threadIdx.y], ~(1 << threadIdx.x));
        }
        __syncwarp(shared.warpActiveLoopThreadMasksPtr[threadIdx.y]);
        if (!loopCondition) break;
        
        uint primeA = d_primes[indexA];
        uint primeB = d_primes[indexB];
        
        uint64_t product = (uint64_t)primeA * primeB;
        uint productModulo = (uint)product;
        
        // Initial rule-out
        bool isModuloProductPrime = true;
        uint givenPrimes[] { 3, 5, 7, 11, 13, 17, 19, 23, 29 };
        for (int i = 0; i < sizeof(givenPrimes) / sizeof(uint) && isModuloProductPrime; i++) {
            isModuloProductPrime = ((productModulo % givenPrimes[i]) != 0);
        }
        
        // Binary search for prime
        if (isModuloProductPrime) {
            int left = 0; // 0;
            int right = primeCount;
            while (left < right) {
                int mid = left + (right - left) / 2;
                if (d_primes[mid] < productModulo) left = mid + 1;
                else right = mid;
            }
            
            isModuloProductPrime = (left < primeCount && d_primes[left] == productModulo);
        }
        
        // Sync threads in the warp, figuring out how many found valid pairs were found that need to be added to the global list.
        int warpThreadsWithValidPairs = __ballot_sync(shared.warpActiveLoopThreadMasksPtr[threadIdx.y], isModuloProductPrime);
        if (isModuloProductPrime) {
        
            // Number of entries to allocate in the global array is equal to the number of threads in the warp that found valid prime/multiplier pairs.
            int warpThreadsWithValidPairsCount = __popc(warpThreadsWithValidPairs);
            
            // Choose one warp thread inside this condition and use it to make the allocation, then read that thread's value in all threads.
            int chosenWarpThread = __ffs(warpThreadsWithValidPairs) - 1;
            int validPairIndexBase = 0;
            if (threadIdx.x == chosenWarpThread) {
                validPairIndexBase = atomicAdd(&d_vars->validPairCount, warpThreadsWithValidPairsCount);
                if (validPairIndexBase + warpThreadsWithValidPairsCount >= VALID_PAIR_COUNT_BUFFER_SIZE) {
                    // We need this value to continue to represent the actual valid count in the buffer.
                    // Race conditions between this and the preceding atomicAdd are inconsequential,
                    // as they could only possibly delay single warp allocations into a next round we already need anyway.
                    atomicSub(&d_vars->validPairCount, warpThreadsWithValidPairsCount);
                }
            }
            validPairIndexBase = __shfl_sync(warpThreadsWithValidPairs, validPairIndexBase, chosenWarpThread);
            
            // On all threads this time, break by continuing without increment and invalidating the index for the loop condition.
            if (validPairIndexBase + warpThreadsWithValidPairsCount >= VALID_PAIR_COUNT_BUFFER_SIZE) {
                indexA = primeCount;
                d_vars->stoppedShort = true;
                continue;
            }
            
            // The relative index is equal to the number of threads before the current which also had valid prime/multiplier pairs.
            int relativeIndex = __popc(warpThreadsWithValidPairs & ((1 << threadIdx.x) - 1));
            
            d_validPairs[validPairIndexBase + relativeIndex] = pair<uint> { primeA, primeB };
        }
        
		indexB++;
		if (indexB >= indexA) {
			indexB = 0;
			indexA++;
		}
        iWithinBatch++;
    }
    
    d_threadIndicesWithinBatch[iBatch] = iWithinBatch;
}

std::vector<pair<uint>> getValidPrimeMultiplierPairs(cudaDeviceProp& deviceProp, std::vector<uint>& primesNarrowed) {

    const std::string filename = "primeMultiplierPairs_" + std::to_string(MAX_SCORED_PRIME_COUNT) + "_" + std::to_string(PRIME_LOWER_BOUND) + "_" + std::to_string(HASH_RANGE_1D) + ".bin";
    std::vector<pair<uint>> validPairs;
    
    if (!fileExists(filename)) {
    
        int blockCount = 0;
        int threadsPerBlock = 0;
        {
            auto status = cudaOccupancyMaxPotentialBlockSize(&blockCount, &threadsPerBlock, findValidPrimeMultiplierPairs, 0, 0);
            if (status != cudaSuccess) {
                std::cout << "CUDA block size query error: " << cudaGetErrorString(status) << std::endl;
            }
        }

        int warpsPerBlock = threadsPerBlock / WARP_SIZE;
        size_t sharedMemorySizePerBlock = (size_t)findValidPrimeMultiplierPairs_sharedMemoryPointers::allocate((uint8_t*)0, warpsPerBlock).endPtr;
        
        int primeCount = primesNarrowed.size();
        int64_t pairCount = (int64_t)primeCount * (primeCount - 1) / 2;
        int threadCount = threadsPerBlock * blockCount;
        int64_t batchSize = (pairCount + threadCount - 1) / threadCount;
        int64_t totalCoverage = (int64_t)batchSize * threadCount;
        
        std::cout << "Total pair count: " << pairCount << std::endl;
        std::cout << "CUDA grid size in blocks: " << blockCount << std::endl;
        std::cout << "CUDA block size in threads: " << threadsPerBlock << std::endl;
        std::cout << "CUDA warp size in threads: " << deviceProp.warpSize << " (expected: " << WARP_SIZE << ")" << std::endl;
        std::cout << "CUDA warps per block: " << warpsPerBlock << std::endl;
        std::cout << "Shared memory size per block: " << sharedMemorySizePerBlock << std::endl;
        std::cout << "CUDA thread count: " << blockCount << " * " << threadsPerBlock << " = " << threadCount << std::endl;
        std::cout << "Batch size: " << batchSize << std::endl;
        std::cout << "Total pair index coverage: " << totalCoverage << std::endl;
        
        uint* d_primes;
        cudaMalloc((void**)&d_primes, primesNarrowed.size() * sizeof(uint));
        cudaMemcpy(d_primes, primesNarrowed.data(), primesNarrowed.size() * sizeof(uint), cudaMemcpyHostToDevice);
        
        int64_t* d_threadIndicesWithinBatch;
        std::vector<int64_t> threadIndicesWithinBatch(threadCount);
        cudaMalloc((void**)&d_threadIndicesWithinBatch, threadCount * sizeof(int64_t));
        cudaMemset(d_threadIndicesWithinBatch, 0, threadCount * sizeof(int64_t));
        
        findValidPrimeMultiplierPairs_vars vars;
        findValidPrimeMultiplierPairs_vars* d_vars;
        cudaMalloc((void**)&d_vars, sizeof(findValidPrimeMultiplierPairs_vars));
        
        pair<uint>* d_validPairs;
        cudaMalloc((void**)&d_validPairs, VALID_PAIR_COUNT_BUFFER_SIZE * sizeof(pair<int>));
        
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        do {
            vars.validPairCount = 0;
            vars.stoppedShort = false;
            cudaMemcpy(d_vars, &vars, sizeof(vars), cudaMemcpyHostToDevice);
            
            cudaEventRecord(start, 0);
            
            std::cout << "Running pair discovery kernel iteration..." << std::endl;
            findValidPrimeMultiplierPairs<<<blockCount, dim3(WARP_SIZE, warpsPerBlock), sharedMemorySizePerBlock>>>(
                d_primes, primesNarrowed.size(), batchSize, d_threadIndicesWithinBatch, d_vars, d_validPairs);
            
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time, start, stop);
            printf("Kernel elapsed time: %.3f s \n", time / (1000));
            
            cudaDeviceSynchronize();
            {
                auto status = cudaGetLastError();
                if (status != cudaSuccess) {
                    std::cout << "CUDA kernel execution error: " << cudaGetErrorString(status) << std::endl;
                    exit(-1);
                }
            }
            
            cudaMemcpy(&vars, d_vars, sizeof(findValidPrimeMultiplierPairs_vars), cudaMemcpyDeviceToHost);
            
            int validPairsOldSize = validPairs.size();
            validPairs.resize(validPairsOldSize + vars.validPairCount);
            assert(vars.validPairCount <= VALID_PAIR_COUNT_BUFFER_SIZE);
            cudaMemcpy(validPairs.data() + validPairsOldSize, d_validPairs, vars.validPairCount * sizeof(pair<uint>), cudaMemcpyDeviceToHost);
            
            cudaMemcpy(threadIndicesWithinBatch.data(), d_threadIndicesWithinBatch, threadCount * sizeof(int), cudaMemcpyDeviceToHost);
            int64_t progress = 0;
            for (int i = 0; i < threadCount; i++) {
                progress += threadIndicesWithinBatch[i];
            }
            double progressPercent = progress * 100.0 / pairCount;
            std::cout << "Pairs found so far: " << validPairs.size() << "; Progress: " << progress << " / " << pairCount << " (" << progressPercent << "%)" << std::endl;
            
            if (vars.stoppedShort != (progress != pairCount)) {
                std::cout << "Warning: `vars.stoppedShort` was " << std::to_string(vars.stoppedShort) << " but `progress` (" << std::to_string(progress) << ") " <<
                    (progress != pairCount ? "!" : "=") << "= `pairCount` (" << std::to_string(pairCount) << ")" << std::endl;
            }
            
        } while (vars.stoppedShort);
        
        std::cout << "Finished pair discovery." << std::endl;
        
        cudaFree(d_validPairs);
        cudaFree(d_vars);
        cudaFree(d_threadIndicesWithinBatch);
        cudaFree(d_primes);
        
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(validPairs.data()), validPairs.size() * sizeof(pair<uint>));
            outFile.close();
            std::cout << "Wrote narrowed primes to file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for writing: " << filename << std::endl;
        }
    } else {
        std::ifstream inFile(filename, std::ios::binary);
        if (inFile.is_open()) {
            inFile.seekg(0, std::ios::end);
            size_t fileSize = inFile.tellg();
            inFile.seekg(0, std::ios::beg);
            
            size_t vectorSize = fileSize / sizeof(pair<uint>);
            validPairs.resize(vectorSize);
            inFile.read(reinterpret_cast<char*>(validPairs.data()), fileSize);
            inFile.close();
            std::cout << "Loaded " << validPairs.size() << " prime multiplier pair" << (validPairs.size() == 1 ? "" : "s") << " from file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for reading: " << filename << std::endl;
        }
    }
        
    return validPairs;
}

/*
 * 2D hash scoring
 */

struct rankPrimePairs_sharedMemoryPointers {
    uint* warpActiveLoopThreadMasksPtr;
    uint8_t* endPtr;
    
    __host__ __device__ static rankPrimePairs_sharedMemoryPointers allocate(uint8_t* sharedPtr, int warpsPerBlock) {
        uint8_t* incrementingSharedPtr = sharedPtr;
        auto allocate = [&incrementingSharedPtr](size_t size) {
            uint8_t* allocatedPtr = incrementingSharedPtr;
            incrementingSharedPtr += size;
            return allocatedPtr;
        };
        rankPrimePairs_sharedMemoryPointers value;
        value.warpActiveLoopThreadMasksPtr = (uint*)allocate(sizeof(int) * warpsPerBlock);
        value.endPtr = incrementingSharedPtr;
        return value;
    }
};

struct multiplierGroup {
    size_t primesIndexBase;
    size_t scoresIndexBase;
    int multiplier;
    int primeCount;
};

struct batchStartIndex {
    size_t multiplierGroupIndex;
    int pairIndexWithin;
};

struct primeMultiplierPairRanking {
    std::vector<multiplierGroup> multiplierGroups;
    std::vector<uint> primesForMultiplierGroups;
    std::vector<float> scores;
};

__global__ void rankPrimePairsKernel(uint* d_primesForMultiplierGroups, multiplierGroup* d_multiplierGroups, int multiplierGroupCount, batchStartIndex* d_threadStartIndices, size_t batchSize, size_t batchSizePrintStart, size_t batchSizePrintTotal, float* d_scores, int scoreLimit) {
	using complex_type = FFT_B::value_type;
    
    extern __shared__ uint8_t sharedMemoryStart[];
    rankPrimePairs_sharedMemoryPointers shared = rankPrimePairs_sharedMemoryPointers::allocate(&sharedMemoryStart[0], blockDim.y);
    
    if (threadIdx.x == 0) {
        shared.warpActiveLoopThreadMasksPtr[threadIdx.y] = 0xFFFFFFFF;
    }
    
    __syncthreads();

    int iThread = threadIdx.y * blockDim.x + threadIdx.x;
    int iBatch = blockIdx.x * (blockDim.x * blockDim.y) + iThread;
    size_t iScoresStart = iBatch * batchSize;
    auto start = d_threadStartIndices[iBatch];
    auto pairIndices = extractIndexPair(start.pairIndexWithin);
	
	complex_type buffer[BIT_RANGE * HASH_RANGE_2D * HASH_RANGE_2D];
	complex_type buffer_local_double[HASH_RANGE_2D * 2];
	
    multiplierGroup multiplierGroupCurrent = d_multiplierGroups[start.multiplierGroupIndex];
    size_t iWithinBatch = 0;
    for (int indexA = pairIndices.a, indexB = pairIndices.b, pairIndex = start.pairIndexWithin, multiplierGroupIndex = start.multiplierGroupIndex;;) {
        bool loopCondition = (iWithinBatch < batchSize && iScoresStart + iWithinBatch < scoreLimit && multiplierGroupIndex < multiplierGroupCount);
        if (!loopCondition) {
            atomicAnd(&shared.warpActiveLoopThreadMasksPtr[threadIdx.y], ~(1 << threadIdx.x));
        }
        __syncwarp(shared.warpActiveLoopThreadMasksPtr[threadIdx.y]);
        if (!loopCondition) break;
        
		int primeA = d_primesForMultiplierGroups[multiplierGroupCurrent.primesIndexBase + indexA];
		int primeB = d_primesForMultiplierGroups[multiplierGroupCurrent.primesIndexBase + indexB];
		
		int xorBase = primeA * -(HASH_RANGE_2D / 2);
		for (int xBufferIndex = 0; xBufferIndex < HASH_RANGE_2D; xBufferIndex++) {
			int bufferIndexWithX = xBufferIndex * (HASH_RANGE_2D * BIT_RANGE);
			int xorAdd = primeB * -(HASH_RANGE_2D / 2);
		    for (int yBufferIndex = 0; yBufferIndex < HASH_RANGE_2D; yBufferIndex++) {
				int bufferIndexWithXY = yBufferIndex * BIT_RANGE + bufferIndexWithX;
				int hash = xorBase ^ xorAdd;
				hash *= multiplierGroupCurrent.multiplier;
				hash ^= (hash >> 15);
				int hashShifted = hash;
				for (int bitIndex = 0; bitIndex < BIT_RANGE; bitIndex++) {
					int bufferIndexWithXYB = bitIndex + bufferIndexWithXY;
					float value = (hashShifted & 1) ? -1.0f : 1.0f;
					buffer[bufferIndexWithXYB] = complex_type{ value, 0.0f };
					hashShifted >>= 1;
				}
				xorAdd += primeB;
				
				// Compute the Discrete Fourier Transform over the bit string we just wrote at this X,Y position.
				FFT_B().execute(&buffer[bufferIndexWithXY]);
			}
			xorBase += primeA;
			
			// At this X position, compute the Discrete Fourier Transform over the Y direction for all bit positions.
			for (int bitIndex = 0; bitIndex < BIT_RANGE; bitIndex++) {
				int bufferIndexWithXB = bitIndex + bufferIndexWithX;
				for (int yBufferIndex = 0; yBufferIndex < HASH_RANGE_2D; yBufferIndex++) buffer_local_double[yBufferIndex] = buffer[bufferIndexWithXB + yBufferIndex * BIT_RANGE];
                for (int yBufferIndex = HASH_RANGE_2D; yBufferIndex < HASH_RANGE_2D * 2; yBufferIndex++) buffer_local_double[yBufferIndex] = complex_type{ 0.0, 0.0 };
				FFT_XY().execute(&buffer_local_double[0]);
				for (int yBufferIndex = 0; yBufferIndex < HASH_RANGE_2D; yBufferIndex++) buffer[bufferIndexWithXB + yBufferIndex * BIT_RANGE] = buffer_local_double[yBufferIndex];
			}
		}
		
		// Compute the Discrete Fourier Transform over the X direction for all Y and bit positions.
        // Replace with enhanced Fourier analyzer for better coherent noise properties
        enhancedFourierAnalyzer analyzer;
        indexDistributionAnalyzer gradientAnalyzer;
        
		for (int yBufferIndex = 0; yBufferIndex < HASH_RANGE_2D; yBufferIndex++) {
			int bufferIndexWithY = yBufferIndex * BIT_RANGE;
			for (int bitIndex = 0; bitIndex < BIT_RANGE; bitIndex++) {
				int bufferIndexWithYB = bitIndex + bufferIndexWithY;
				for (int xBufferIndex = 0; xBufferIndex < HASH_RANGE_2D; xBufferIndex++) buffer_local_double[xBufferIndex] = buffer[bufferIndexWithYB + xBufferIndex * (HASH_RANGE_2D * BIT_RANGE)];
                for (int xBufferIndex = HASH_RANGE_2D; xBufferIndex < HASH_RANGE_2D * 2; xBufferIndex++) buffer_local_double[xBufferIndex] = complex_type{ 0.0, 0.0 };
				FFT_XY().execute(&buffer_local_double[0]);
                
				for (int xBufferIndex = 0; xBufferIndex < HASH_RANGE_2D; xBufferIndex++) {
                    complex_type& value = buffer_local_double[xBufferIndex];
                    
                    // Convert buffer indices to frequency values
                    // For FFT, frequencies are in range [-N/2, N/2-1] after reordering
                    int xFreq = xBufferIndex;
                    if (xFreq >= HASH_RANGE_2D / 2) xFreq -= HASH_RANGE_2D;
                    
                    int yFreq = yBufferIndex;
                    if (yFreq >= HASH_RANGE_2D / 2) yFreq -= HASH_RANGE_2D;
                    
                    // Pass frequency information for better analysis
                    analyzer.accept(value.x, value.y, xFreq, yFreq, HASH_RANGE_2D/2);
                }
			}
		}
        
        // Also analyze gradient distribution by simulating how these primes would generate gradients
        int hashBase = primeA * primeB * multiplierGroupCurrent.multiplier;
        for (int i = 0; i < 64; i++) {  // Sample some hash values
            int hash = hashBase + i;
            hash ^= (hash >> 15);
            gradientAnalyzer.analyzeHash(hash);
        }
        
        if (batchSizePrintTotal > 0 && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
            size_t progress = iWithinBatch + batchSizePrintStart + 1;
            printf("[Thread 0] Progress: %llu / %llu (%.4f%)\n", progress, batchSizePrintTotal, progress * 100.0 / batchSizePrintTotal);
        }
		
        // Compute combined score with components for coherent noise quality
        double spectralScore = analyzer.compute();
        double gradientScore = gradientAnalyzer.computeUniformity();
        
        // Combine scores - spectral properties (80%) and gradient distribution (20%)
        double score = 0.8 * spectralScore + 0.2 * gradientScore;
        
        // Debugging info for first few calculations (if in thread 0)
        if (iWithinBatch < 5 && threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0) {
            printf("Debug [%llu] - Spectral: %.4f, Gradient: %.4f, Combined: %.4f\n", 
                   iWithinBatch, spectralScore, gradientScore, score);
        }
        
        d_scores[iScoresStart + iWithinBatch] = (float)score;
                        
		// Next pair of primes
		indexB++;
		if (indexB >= indexA) {
			indexB = 0;
			indexA++;
		}
        pairIndex++;
        if (indexA >= multiplierGroupCurrent.primeCount) {
            indexA = indexB = pairIndex = 0;
            multiplierGroupIndex++;
            if (multiplierGroupIndex < multiplierGroupCount) {
                multiplierGroupCurrent = d_multiplierGroups[multiplierGroupIndex];
            }
        }
        iWithinBatch++;
	}
}

size_t populateThreadStartIndices(std::vector<batchStartIndex>& threadStartIndices, size_t& iMultiplierGroup, std::vector<multiplierGroup>& multiplierGroups, size_t cumulativePairCountThisIteration, size_t cumulativePairCountResumePoint) {
    int threadCount = threadStartIndices.size();
    size_t batchSizeThisIteration = (cumulativePairCountThisIteration + threadCount - 1) / threadCount;
    
    int iThread = 0;
    // Make sure we don't go out of bounds
    if (iMultiplierGroup >= multiplierGroups.size()) {
        for (iThread = 0; iThread < threadCount; iThread++) {
            threadStartIndices[iThread] = { multiplierGroups.size(), 0 };
        }
        return batchSizeThisIteration;
    }
    
    multiplierGroup multiplierGroup = multiplierGroups[iMultiplierGroup];
    int pairCount = multiplierGroup.primeCount * (multiplierGroup.primeCount - 1) / 2;
    
    for (; iThread < threadCount; iThread++) {
        size_t threadBatchStartWithinRun = batchSizeThisIteration * iThread;
        if (threadBatchStartWithinRun >= cumulativePairCountThisIteration) break;
        size_t threadBatchStart = threadBatchStartWithinRun + cumulativePairCountResumePoint;
        while (multiplierGroup.scoresIndexBase + pairCount <= threadBatchStart) {
            iMultiplierGroup++;
            if (iMultiplierGroup >= multiplierGroups.size()) break;
            
            multiplierGroup = multiplierGroups[iMultiplierGroup];
            pairCount = multiplierGroup.primeCount * (multiplierGroup.primeCount - 1) / 2;
    
        }
        if (iMultiplierGroup >= multiplierGroups.size()) break;
        
        int pairIndexWithin = (int)(threadBatchStart - multiplierGroup.scoresIndexBase);
        if (pairIndexWithin >= pairCount) break;
        
        threadStartIndices[iThread] = batchStartIndex { iMultiplierGroup, pairIndexWithin };
    }
    
    for (; iThread < threadCount; iThread++) {
        threadStartIndices[iThread] = { multiplierGroups.size(), 0 };
    }
    
    return batchSizeThisIteration;
}

template<bool IsTest = false>
void runRankPrimePairsKernel(size_t pairScoreBufferSize, int cumulativePairCount, std::vector<batchStartIndex>& threadStartIndices, batchStartIndex* d_threadStartIndices, std::vector<multiplierGroup>& multiplierGroups,
        multiplierGroup* d_multiplierGroups, uint* d_primesForMultiplierGroups, std::vector<float>& scores, float* d_scores, int blockCount, int warpsPerBlock, size_t sharedMemorySizePerBlock) {
    int threadCount = threadStartIndices.size();
    size_t iMultiplierGroup = 0;
    size_t batchSizePrintStart = 0;
    size_t batchSizePrintTotal = 0;
    if (!IsTest) {
        for (size_t cumulativePairCountProgress = 0; cumulativePairCountProgress < cumulativePairCount; ) {
            size_t cumulativePairCountThisIteration = std::min(cumulativePairCount - cumulativePairCountProgress, (size_t)PAIR_SCORE_BUFFER_SIZE);
            batchSizePrintTotal += (cumulativePairCountThisIteration + threadCount - 1) / threadCount;
            cumulativePairCountProgress += cumulativePairCountThisIteration;
        }
    }
    for (size_t cumulativePairCountProgress = 0; cumulativePairCountProgress < cumulativePairCount; ) {
        size_t cumulativePairCountThisIteration = std::min(cumulativePairCount - cumulativePairCountProgress, pairScoreBufferSize);
        size_t batchSizeThisIteration = populateThreadStartIndices(threadStartIndices, iMultiplierGroup, multiplierGroups, cumulativePairCountThisIteration, cumulativePairCountProgress);
        size_t totalCoverageThisIteration = batchSizeThisIteration * threadCount;
        
        if (!IsTest) {
            std::cout << "Batch size this iteration: " << batchSizeThisIteration << std::endl;
            std::cout << "Total pair index coverage this iteration: " << totalCoverageThisIteration << std::endl;
        }
        
        cudaMemcpy(d_threadStartIndices, threadStartIndices.data(), threadStartIndices.size() * sizeof(batchStartIndex), cudaMemcpyHostToDevice);
        if (IsTest) cudaMemset(d_scores, 0, cumulativePairCountThisIteration * sizeof(float));
            
        float time;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        
        if (!IsTest) std::cout << "Running pair rank kernel..." << std::endl;
        rankPrimePairsKernel<<<blockCount, dim3(WARP_SIZE, warpsPerBlock), sharedMemorySizePerBlock>>>(
            d_primesForMultiplierGroups, d_multiplierGroups, multiplierGroups.size(), d_threadStartIndices, batchSizeThisIteration, batchSizePrintStart, batchSizePrintTotal, d_scores, cumulativePairCountThisIteration);
            
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        if (!IsTest) printf("Completed. Kernel elapsed time: %.3f s \n", time / 1000);
        
        // When processing chunks, we need to copy to the correct offset in the scores array
        cudaMemcpy(scores.data() + cumulativePairCountProgress, d_scores, cumulativePairCountThisIteration * sizeof(float), cudaMemcpyDeviceToHost);
        
        cumulativePairCountProgress += cumulativePairCountThisIteration;
        batchSizePrintStart += batchSizeThisIteration;
    }
}

primeMultiplierPairRanking rankPrimePairs(cudaDeviceProp& deviceProp, std::vector<pair<uint>>& primeMultiplierPairs) {
    
    std::vector<multiplierGroup> multiplierGroups;
    std::vector<uint> primesForMultiplierGroups;
    std::vector<float> scores;
    
    const std::string filename = "primeMultiplierPairRanking_" + std::to_string(primeMultiplierPairs.size()) + "_" + std::to_string(MAX_SCORED_PRIME_COUNT) + "_" + std::to_string(PRIME_LOWER_BOUND) + "_" + std::to_string(HASH_RANGE_1D) + ".bin";
    std::vector<uint> primesNarrowed;
    primesNarrowed.reserve(MAX_SCORED_PRIME_COUNT);
    
    if (!fileExists(filename)) {
    
        std::vector<pair<uint>> primeMultiplierPairsFull;
        primeMultiplierPairsFull.reserve(primeMultiplierPairs.size() * 2);
        std::transform(primeMultiplierPairs.begin(), primeMultiplierPairs.end(), std::back_inserter(primeMultiplierPairsFull), [](const pair<uint>& entry) { return pair<uint>{ entry.b, entry.a }; });
        std::sort(primeMultiplierPairs.begin(), primeMultiplierPairs.end(), [](const pair<uint>& a, const pair<uint>& b) { return a.a < b.a; });
        
        int lastStartSourceIndex = 0;
        uint lastMultiplier = 0;
        int primeCount = 0;
        size_t cumulativePairCount = 0;
        for (int i = 0; i < primeMultiplierPairs.size(); i++) {
            primeCount++;
            if (primeMultiplierPairs[i].a != lastMultiplier) {
                if (primeCount > 1) {
                    int pairCount = primeCount * (primeCount - 1) / 2;
                    multiplierGroups.push_back(multiplierGroup { primesForMultiplierGroups.size(), cumulativePairCount, (int)lastMultiplier, primeCount });
                    cumulativePairCount += pairCount;
                    for (int i = 0; i < primeCount; i++) {
                        primesForMultiplierGroups.push_back(primeMultiplierPairsFull[lastStartSourceIndex + i].b);
                    }
                }
                lastStartSourceIndex = i;
                lastMultiplier = primeMultiplierPairs[i].a;
                primeCount = 0;
            }
        }
    
        if (multiplierGroups.size() == 0) {
            return { multiplierGroups, primesForMultiplierGroups, scores };
        }

        int blockCount = 0;
        int threadsPerBlock = 0;
        {
            auto status = cudaOccupancyMaxPotentialBlockSize(&blockCount, &threadsPerBlock, rankPrimePairsKernel, 0, 0);
            if (status != cudaSuccess) {
                std::cout << "CUDA block size query error: " << cudaGetErrorString(status) << std::endl;
            }
        }
        
        int warpsPerBlock = threadsPerBlock / WARP_SIZE;
        size_t sharedMemorySizePerBlock = (size_t)rankPrimePairs_sharedMemoryPointers::allocate((uint8_t*)0, warpsPerBlock).endPtr;
        int threadCount = threadsPerBlock * blockCount;
        
        std::cout << "Multiplier groups: " << multiplierGroups.size() << std::endl;
        std::cout << "Total pair count: " << cumulativePairCount << std::endl;
        std::cout << "CUDA grid size in blocks: " << blockCount << std::endl;
        std::cout << "CUDA block size in threads: " << threadsPerBlock << std::endl;
        std::cout << "CUDA warp size in threads: " << deviceProp.warpSize << " (expected: " << WARP_SIZE << ")" << std::endl;
        std::cout << "CUDA warps per block: " << warpsPerBlock << std::endl;
        std::cout << "Shared memory size per block: " << sharedMemorySizePerBlock << std::endl;
        std::cout << "CUDA thread count: " << blockCount << " * " << threadsPerBlock << " = " << threadCount << std::endl;
        
        uint* d_primesForMultiplierGroups;
        size_t primesSize = primesForMultiplierGroups.size() * sizeof(uint);
        std::cout << "Allocating " << (primesSize / (1024.0 * 1024.0)) << " MB for primes array" << std::endl;
        cudaMalloc((void**)&d_primesForMultiplierGroups, primesSize);
        cudaMemcpy(d_primesForMultiplierGroups, primesForMultiplierGroups.data(), primesSize, cudaMemcpyHostToDevice);
        
        multiplierGroup* d_multiplierGroups;
        size_t groupsSize = multiplierGroups.size() * sizeof(multiplierGroup);
        std::cout << "Allocating " << (groupsSize / (1024.0 * 1024.0)) << " MB for multiplier groups array" << std::endl;
        cudaMalloc((void**)&d_multiplierGroups, groupsSize);
        cudaMemcpy(d_multiplierGroups, multiplierGroups.data(), groupsSize, cudaMemcpyHostToDevice);
        
        scores.resize(cumulativePairCount);
        float* d_scores;
        size_t scoresSize = std::min(cumulativePairCount, (size_t)PAIR_SCORE_BUFFER_SIZE) * sizeof(float);
        std::cout << "Allocating " << (scoresSize / (1024.0 * 1024.0)) << " MB for scores array" << std::endl;
        cudaMalloc((void**)&d_scores, scoresSize);
        
        std::vector<batchStartIndex> threadStartIndices(threadCount);
        batchStartIndex* d_threadStartIndices;
        size_t indicesSize = threadStartIndices.size() * sizeof(batchStartIndex);
        std::cout << "Allocating " << (indicesSize / (1024.0 * 1024.0)) << " MB for thread start indices array" << std::endl;
        cudaMalloc((void**)&d_threadStartIndices, indicesSize);
        
        runRankPrimePairsKernel(PAIR_SCORE_BUFFER_SIZE, cumulativePairCount, threadStartIndices, d_threadStartIndices, multiplierGroups, d_multiplierGroups,
            d_primesForMultiplierGroups, scores, d_scores, blockCount, warpsPerBlock, sharedMemorySizePerBlock);
        
        cudaFree(d_scores);
        cudaFree(d_multiplierGroups);
        cudaFree(d_primesForMultiplierGroups);
        cudaFree(d_threadStartIndices);

        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            uint64_t sizes[] { multiplierGroups.size(), primesForMultiplierGroups.size(), scores.size() };
            outFile.write(reinterpret_cast<char*>(&sizes), sizeof(sizes));
            outFile.write(reinterpret_cast<const char*>(multiplierGroups.data()), multiplierGroups.size() * sizeof(multiplierGroup));
            outFile.write(reinterpret_cast<const char*>(primesForMultiplierGroups.data()), primesForMultiplierGroups.size() * sizeof(uint));
            outFile.write(reinterpret_cast<const char*>(scores.data()), scores.size() * sizeof(float));
            outFile.close();
            std::cout << "Wrote scored prime multiplier group data to file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for writing: " << filename << std::endl;
        }
    } else {
        std::ifstream inFile(filename, std::ios::binary);
        if (inFile.is_open()) {
            uint64_t sizes[3];
            inFile.read(reinterpret_cast<char*>(&sizes), sizeof(sizes));
            multiplierGroups.resize(sizes[0]);
            primesForMultiplierGroups.resize(sizes[1]);
            scores.resize(sizes[2]);
            
            inFile.read(reinterpret_cast<char*>(multiplierGroups.data()), multiplierGroups.size() * sizeof(multiplierGroup));
            inFile.read(reinterpret_cast<char*>(primesForMultiplierGroups.data()), primesForMultiplierGroups.size() * sizeof(uint));
            inFile.read(reinterpret_cast<char*>(scores.data()), scores.size() * sizeof(float));
            inFile.close();
            std::cout << "Loaded " << multiplierGroups.size() << " multiplier group" << (multiplierGroups.size() == 1 ? "" : "s");
            std::cout << ", " << primesForMultiplierGroups.size() << " corresponding prime" << (primesForMultiplierGroups.size() == 1 ? "" : "s");
            std::cout << ", and " << scores.size() << " corresponding score" << (scores.size() == 1 ? "" : "s") << " from file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for reading: " << filename << std::endl;
        }
    }
    
    return { multiplierGroups, primesForMultiplierGroups, scores };
}

/*
 * 
 */

struct scoredInteger {
    float score;
    int value;
};

struct indexedScoreMultiplierPairEntry {
    scoredInteger scoreMultiplierPair;
    size_t combinationStartIndex;
};

struct combinationScorecard {
    std::vector<scoredInteger> sortedScoreMultiplierPairs;
    std::vector<uint> sortedPrimeCombinations;
};

__inline__ int countCombinations(int memberCount, int cardinality) {
    int combinationCountHere = 1;
    for (int i = 0; i < cardinality; i++) {
        combinationCountHere *= (memberCount - i);
        combinationCountHere /= (i + 1);
    }
    return combinationCountHere;
}

struct threadBestCombinations {
    std::vector<indexedScoreMultiplierPairEntry> combinations;
    std::vector<uint> primes;
};

combinationScorecard getBestCombinations(primeMultiplierPairRanking& primeMultiplierPairRanking, int cardinality) {
    std::vector<scoredInteger> sortedScoreMultiplierPairs;
    std::vector<uint> sortedPrimeCombinations;
    
    const std::string filename = "finalCombinations_" + std::to_string(cardinality) + "_" + std::to_string(COMBINATIONS_TO_OUTPUT) + "_" + 
        std::to_string(primeMultiplierPairRanking.multiplierGroups.size()) + "_" +
        std::to_string(primeMultiplierPairRanking.primesForMultiplierGroups.size()) + "_" + 
        std::to_string(primeMultiplierPairRanking.scores.size()) + "_" +
        std::to_string(HASH_RANGE_1D) + ".bin";
    
    if (!fileExists(filename)) {       
        size_t combinationCountTotal = 0;
        for (int iGroup = 0; iGroup < primeMultiplierPairRanking.multiplierGroups.size(); iGroup++) {
            multiplierGroup multiplierGroup = primeMultiplierPairRanking.multiplierGroups[iGroup];
            int combinationCountHere = countCombinations(multiplierGroup.primeCount, cardinality);
            combinationCountTotal += combinationCountHere;
        }
        
        std::cout << "Total combinations: " << std::to_string(combinationCountTotal) << " (cardinality: " << cardinality << ")" << std::endl;

        // Create thread pool and results vector
        const int threadCount = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(threadCount);
        std::vector<threadBestCombinations> threadResults(threadCount);
        std::atomic<size_t> nextGroupIndex(0);

        // Launch threads
        for (int iThread = 0; iThread < threadCount; iThread++) {
            threads[iThread] = std::thread([&, iThread]() {
                std::vector<int> groupPrimeIndices(cardinality);
                size_t processedCount = 0;
                
                while (true) {
                    // Get next group to process
                    size_t iGroup = nextGroupIndex.fetch_add(1);
                    if (iGroup >= primeMultiplierPairRanking.multiplierGroups.size()) break;
                    
                    multiplierGroup& multiplierGroup = primeMultiplierPairRanking.multiplierGroups[iGroup];
                    if (multiplierGroup.primeCount < cardinality) continue;

                    for (int i = 0; i < cardinality; i++) groupPrimeIndices[i] = (cardinality - 1 - i);
                    while (groupPrimeIndices[0] < multiplierGroup.primeCount) {
                        double combinationPairScoreSum = 0;
                        for (int iPrimeA = 1; iPrimeA < cardinality; iPrimeA++) {
                            for (int iPrimeB = 0; iPrimeB < iPrimeA; iPrimeB++) {
                                int iPairMapped = (int)mergeIndexPair(groupPrimeIndices[iPrimeA], groupPrimeIndices[iPrimeB]);
                                combinationPairScoreSum += primeMultiplierPairRanking.scores[multiplierGroup.scoresIndexBase + iPairMapped];
                            }
                        }

                        // Store combination if it's among thread's best
                        if (threadResults[iThread].combinations.size() < COMBINATIONS_TO_OUTPUT || 
                            combinationPairScoreSum < threadResults[iThread].combinations[0].scoreMultiplierPair.score) {
                            std::vector<uint> primesInCombination(cardinality);
                            for (int i = 0; i < cardinality; i++) {
                                primesInCombination[i] = primeMultiplierPairRanking.primesForMultiplierGroups[
                                    multiplierGroup.primesIndexBase + groupPrimeIndices[i]];
                            }

                            indexedScoreMultiplierPairEntry entry{
                                scoredInteger{ (float)combinationPairScoreSum, multiplierGroup.multiplier },
                                processedCount
                            };

                            if (threadResults[iThread].combinations.size() < COMBINATIONS_TO_OUTPUT) {
                                threadResults[iThread].combinations.push_back(entry);
                                std::push_heap(threadResults[iThread].combinations.begin(), threadResults[iThread].combinations.end(),
                                    [](const auto& a, const auto& b) { return a.scoreMultiplierPair.score < b.scoreMultiplierPair.score; });
                            } else {
                                std::pop_heap(threadResults[iThread].combinations.begin(), threadResults[iThread].combinations.end(),
                                    [](const auto& a, const auto& b) { return a.scoreMultiplierPair.score < b.scoreMultiplierPair.score; });
                                threadResults[iThread].combinations.back() = entry;
                                std::push_heap(threadResults[iThread].combinations.begin(), threadResults[iThread].combinations.end(),
                                    [](const auto& a, const auto& b) { return a.scoreMultiplierPair.score < b.scoreMultiplierPair.score; });
                            }

                            // Store the primes for this combination
                            if (threadResults[iThread].primes.size() < COMBINATIONS_TO_OUTPUT * cardinality) {
                                threadResults[iThread].primes.insert(threadResults[iThread].primes.end(), 
                                    primesInCombination.begin(), primesInCombination.end());
                            }
                        }

                        processedCount++;
                        // if (processedCount % 100000000 == 0) {
                        //     std::lock_guard<std::mutex> lock(printMutex);
                        //     double percentage = (double)processedCount / (double)combinationCountTotal * 100.0;
                        //     std::cout << "Thread " << iThread << " processed " << processedCount << 
                        //         " combinations (" << percentage << "%)..." << std::endl;
                        // }

                        // Next indices
                        int lastIndex = -1;
                        for (int i = cardinality - 1;; i--) {
                            groupPrimeIndices[i]++;
                            if (i <= 0 || groupPrimeIndices[i] < groupPrimeIndices[i - 1]) break;
                            else groupPrimeIndices[i] = lastIndex + 1;
                            lastIndex = groupPrimeIndices[i];
                        }
                    }

                    if (iGroup % 10000 == 0) {
                        double percentage = (double)iGroup / (double)primeMultiplierPairRanking.multiplierGroups.size() * 100.0;
                        std::cout << "Thread " << iThread << " processed group " << iGroup << 
                            " (" << percentage << "%)..." << std::endl;
                    }
                }
            });
        }

        // Wait for all threads to complete
        for (auto& thread : threads) {
            thread.join();
        }

        // Merge results from all threads
        std::vector<indexedScoreMultiplierPairEntry> mergedCombinations;
        for (const auto& threadResult : threadResults) {
            mergedCombinations.insert(mergedCombinations.end(), 
                threadResult.combinations.begin(), threadResult.combinations.end());
        }

        // Sort and keep best COMBINATIONS_TO_OUTPUT
        std::sort(mergedCombinations.begin(), mergedCombinations.end(),
            [](const auto& a, const auto& b) { return a.scoreMultiplierPair.score < b.scoreMultiplierPair.score; });
        if (mergedCombinations.size() > COMBINATIONS_TO_OUTPUT) {
            mergedCombinations.resize(COMBINATIONS_TO_OUTPUT);
        }

        // Extract final sorted results
        sortedScoreMultiplierPairs.reserve(mergedCombinations.size());
        for (const auto& entry : mergedCombinations) {
            sortedScoreMultiplierPairs.push_back(entry.scoreMultiplierPair);
        }

        // Merge prime combinations in the same order
        for (const auto& threadResult : threadResults) {
            sortedPrimeCombinations.insert(sortedPrimeCombinations.end(),
                threadResult.primes.begin(), threadResult.primes.end());
        }
        if (sortedPrimeCombinations.size() > COMBINATIONS_TO_OUTPUT * cardinality) {
            sortedPrimeCombinations.resize(COMBINATIONS_TO_OUTPUT * cardinality);
        }

        // Save results
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            uint64_t size = (uint64_t)sortedScoreMultiplierPairs.size();
            outFile.write(reinterpret_cast<char*>(&size), sizeof(uint64_t));
            outFile.write(reinterpret_cast<const char*>(sortedScoreMultiplierPairs.data()), 
                sortedScoreMultiplierPairs.size() * sizeof(scoredInteger));
            outFile.write(reinterpret_cast<const char*>(sortedPrimeCombinations.data()),
                sortedPrimeCombinations.size() * sizeof(uint));
            outFile.close();
            std::cout << "Wrote top " << COMBINATIONS_TO_OUTPUT << " combinations to file: " << filename << std::endl;
        }
    } else {
        std::ifstream inFile(filename, std::ios::binary);
        if (inFile.is_open()) {
            uint64_t size;
            inFile.read(reinterpret_cast<char*>(&size), sizeof(uint64_t));
            
            sortedScoreMultiplierPairs.resize(size);
            sortedPrimeCombinations.resize(size * cardinality);
            
            inFile.read(reinterpret_cast<char*>(sortedScoreMultiplierPairs.data()), 
                size * sizeof(scoredInteger));
            inFile.read(reinterpret_cast<char*>(sortedPrimeCombinations.data()),
                size * cardinality * sizeof(uint));
                
            inFile.close();
            std::cout << "Loaded " << size << " combinations from file: " << filename << std::endl;
        } else {
            std::cout << "Error opening file for reading: " << filename << std::endl;
        }
    }
    
    return combinationScorecard { sortedScoreMultiplierPairs, sortedPrimeCombinations };
}

/*
 * Main
 */

#if EXECUTE_TEST_RUN
void test();
#endif

// Visualization helper function to evaluate coherent noise quality
void visualizeNoiseAndSpectrum(const std::vector<scoredInteger>& bestScoreMultiplierPairs, 
                              const std::vector<uint>& bestPrimeCombinations, 
                              int cardinality) {
    // Create a "visualization" directory if it doesn't exist
    std::string dirName = "visualization";
    if (!std::filesystem::exists(dirName)) {
        std::filesystem::create_directory(dirName);
    }
    
    // Output information about top combinations
    std::ofstream infoFile(dirName + "/combinations_info.txt");
    if (!infoFile) {
        std::cout << "Could not open info file for writing" << std::endl;
        return;
    }
    
    // Write information about the top combinations
    infoFile << "Top " << std::min(10, (int)bestScoreMultiplierPairs.size()) << " prime combinations for coherent noise:" << std::endl;
    
    for (int i = 0; i < std::min(10, (int)bestScoreMultiplierPairs.size()); i++) {
        const auto& entry = bestScoreMultiplierPairs[i];
        infoFile << "Combination " << i + 1 << ":" << std::endl;
        infoFile << "  Score: " << entry.score << std::endl;
        infoFile << "  Multiplier: 0x" << std::hex << entry.value << std::dec << std::endl;
        infoFile << "  Primes: ";
        
        for (int j = 0; j < cardinality; j++) {
            uint prime = bestPrimeCombinations[i * cardinality + j];
            infoFile << "0x" << std::hex << prime << std::dec;
            if (j < cardinality - 1) infoFile << ", ";
        }
        infoFile << std::endl;
        
        // Generate a simple ASCII visualization of noise characteristics
        infoFile << std::endl << "Noise Pattern Preview (ASCII):" << std::endl;
        const int width = 40;
        const int height = 20;
        
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int hash = 0;
                uint multiplier = entry.value;
                
                // Use all primes to generate hash
                for (int j = 0; j < cardinality; j++) {
                    uint prime = bestPrimeCombinations[i * cardinality + j];
                    hash ^= (int)(x * prime) ^ (int)(y * (prime >> 8));
                }
                
                hash *= multiplier;
                hash ^= (hash >> 15);
                
                // Convert hash to a simple gradient value
                float noiseVal = (hash & 0xFFFF) / (float)0xFFFF;
                
                // Map to ASCII character based on value
                const char* gradientChars = " .:-=+*#%@";
                int charIndex = (int)(noiseVal * 9);
                infoFile << gradientChars[charIndex];
            }
            infoFile << std::endl;
        }
        
        infoFile << std::endl << "-----------------------------------------------------" << std::endl << std::endl;
    }
    
    infoFile.close();
    std::cout << "Wrote visualization data to " << dirName << "/combinations_info.txt" << std::endl;
    
    // For actual applications, you would want to output proper image files 
    // with noise fields and FFT visualizations, but that's beyond the scope
    // of this simple preview implementation
}

int main()
{
    int cudeDev = 0;
    cudaSetDevice(cudeDev);
	
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, cudeDev);
    std::cout << "Using CUDA Device: " << deviceProp.name << std::endl;

#if EXECUTE_TEST_RUN
    test();
#endif

    std::vector<uint> primes = getPrimes();
    primes = narrowPrimes(deviceProp, primes);
    std::vector<pair<uint>> validPrimeMultiplierPairs = getValidPrimeMultiplierPairs(deviceProp, primes);
    primeMultiplierPairRanking primeMultiplierPairRanking = rankPrimePairs(deviceProp, validPrimeMultiplierPairs);
    combinationScorecard combinationScorecard = getBestCombinations(primeMultiplierPairRanking, TARGET_PRIME_GROUP_SIZE);
    std::cout << "Total combinations: " << combinationScorecard.sortedScoreMultiplierPairs.size() << std::endl;

    // Find the smallest prime in all combinations
    uint smallestPrime = std::numeric_limits<uint>::max();
    
    // Open a file to write the results
    std::ofstream resultsFile("results.txt");
    resultsFile << "Total combinations: " << combinationScorecard.sortedScoreMultiplierPairs.size() << std::endl;
    
    for (size_t iEntry = 0; iEntry < combinationScorecard.sortedScoreMultiplierPairs.size(); iEntry++) {
        scoredInteger& scoreMultiplierPair = combinationScorecard.sortedScoreMultiplierPairs[iEntry];
        std::cout << std::uppercase << std::hex << std::setw(8) << std::setfill('0');
        std::cout << std::endl;
        std::cout << "Multiplier: 0x" << scoreMultiplierPair.value << "; Score: " << scoreMultiplierPair.score << std::endl;
        std::cout << "Primes:";
        
        // Also write to file
        resultsFile << std::uppercase << std::hex << std::setw(8) << std::setfill('0');
        resultsFile << std::endl;
        resultsFile << "Multiplier: 0x" << scoreMultiplierPair.value << "; Score: " << scoreMultiplierPair.score << std::endl;
        resultsFile << "Primes:";
        
        for (int i = 0; i < TARGET_PRIME_GROUP_SIZE; i++) {
            uint currentPrime = combinationScorecard.sortedPrimeCombinations[iEntry * TARGET_PRIME_GROUP_SIZE + i];
            std::cout << " 0x" << currentPrime;
            resultsFile << " 0x" << currentPrime;
            if (currentPrime < smallestPrime) {
                smallestPrime = currentPrime;
            }
        }
        std::cout << std::endl;
        resultsFile << std::endl;
    }

    std::cout << "Smallest prime: 0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << smallestPrime << std::endl;
    resultsFile << "Smallest prime: 0x" << std::uppercase << std::hex << std::setw(8) << std::setfill('0') << smallestPrime << std::endl;
    
    resultsFile.close();
    std::cout << "Results written to results.txt" << std::endl;

    // Visualize the top combinations
    //visualizeNoiseAndSpectrum(combinationScorecard.sortedScoreMultiplierPairs, combinationScorecard.sortedPrimeCombinations, TARGET_PRIME_GROUP_SIZE);

    return 0;
}

/*
 * Tests
 */

#if EXECUTE_TEST_RUN

template<typename TFloat>
bool almostEqual(TFloat a, TFloat b, int bits) {
    return std::min(a, b) / std::max(a, b) > (double)((1 << bits) - 1) / (1 << bits);
}

void testIndicesExhaustive() {
    const std::string testPrefix = "[TEST INDICES EXHAUSTIVE] ";
    constexpr int TOTAL = 512;
    constexpr int PAIR_COUNT = TOTAL * (TOTAL - 1) / 2;
    int successCount = 0;
    for (int iPair = 0; iPair < PAIR_COUNT; iPair++) {
        auto indices = extractIndexPair(iPair);
        int iPairReconstructed = mergeIndexPair(indices.a, indices.b);
        if (iPairReconstructed != iPair) {
            std::cout << testPrefix << "Error: " << iPair << " => (" + indices.a << ", " << indices.b << ") => " << iPairReconstructed << std::endl;
        } else {
            successCount++;
        }
    }
    std::cout << testPrefix << successCount << " / " << PAIR_COUNT << std::endl;
}

void testHeap() {
    constexpr int nTopResults = 512;
    const std::string testPrefix = "[TEST HEAP RANDOM] ";
    std::srand(1);
	heapEntry<float, pair<int>> finalPairs[nTopResults];
	std::fill_n(finalPairs, nTopResults, heapEntry<float, pair<int>>{ INFINITY, pair<int>{ 0, 0 } });
    for (int i = 0; i < nTopResults; i++) {
        float score = std::rand();
        heapReplaceSiftDown<float, pair<int>, nTopResults, 1>(score, pair<int>{ 0, 0 }, &finalPairs[0]);
    }
    int heapCheckCountTotal = 0;
    int heapCheckCountSuccess = 0;
    for (int i = 1; i <= nTopResults; i++) {
        if (i * 2 - 1 >= nTopResults) break;
        heapCheckCountTotal++;
        if (finalPairs[i - 1].key < finalPairs[i * 2 - 1].key) {
            std::cout << testPrefix << "Error: heapOneIndexed[" << i << "].key < heapOneIndexed[" << (i * 2) << "].key" << std::endl;
        } else {
            heapCheckCountSuccess++;
        }
        if (i * 2 >= nTopResults) break;
        heapCheckCountTotal++;
        if (finalPairs[i - 1].key < finalPairs[i * 2].key) {
            std::cout << testPrefix << "Error: heapOneIndexed[" << i << "].key < heapOneIndexed[" << (i * 2 + 1) << "].key" << std::endl;
        } else {
            heapCheckCountSuccess++;
        }
    }
    std::cout << testPrefix << heapCheckCountSuccess << " / " << heapCheckCountTotal << std::endl;
    for (int i = 0; i < nTopResults; i++) {
        if (finalPairs[i].key == INFINITY) {
            std::cout << testPrefix << "Error: heap[" << i << "] is still the default value." << std::endl;
        }
    }
}

void testFourierScorer() {
    const std::string testPrefix = "[TEST FOURIER SCORE EQUIVALENCE] ";
    constexpr size_t size = 32 * 32 * 31;
    std::srand(33333);
    
    testIndicesExhaustive();
    testHeap();
    pair<float> buffer[size];
    double sumOfSquares = 0;
    fourierSpectrumScorer scorer;
    for (int i = 0; i < size; i++) {
        float magnitude = std::max(std::rand(), std::rand()) * (1.0f / RAND_MAX);
        float angle = std::rand() * (6.283185307179586f / RAND_MAX);
        buffer[i] = pair<float> { sin(angle) * magnitude, cos(angle) * magnitude };
        sumOfSquares += magnitude * magnitude;
        scorer.accept(buffer[i].a, buffer[i].b);
    }
    double averageOfSquares = sumOfSquares / size;
    double sumOfNormalizedSquaredDeltas = 0;
    for (int i = 0; i < size; i++) {
        float normalizedDelta = (buffer[i].a * buffer[i].a + buffer[i].b * buffer[i].b) / averageOfSquares - 1;
        sumOfNormalizedSquaredDeltas += normalizedDelta * normalizedDelta;
    }
    double scoreA = ((sumOfNormalizedSquaredDeltas / size) + 1) / size;
    double scoreB = scorer.compute();
    
    assert(almostEqual(scoreA, scoreB, 8));
    std::cout << testPrefix << scoreA << " ~ " << scoreB << std::endl;
}

void testRankPrimePairsKernel() {
    const std::string testPrefix = "[TEST RANK KERNEL] ";
    constexpr int multiplierGroupCount = 32;
    constexpr int combinationCardinality = 4;
    constexpr size_t pairScoreBufferSize = 1 << 12;
    std::srand(111);
    
    std::vector<multiplierGroup> multiplierGroups(multiplierGroupCount);
    std::vector<uint> primesForMultiplierGroups;
    std::vector<float> scores;
    std::vector<float> scores2;
    
    size_t cumulativePairCount = 0;
    for (int i = 0; i < multiplierGroupCount; i++) {
        int primeCountHere = ((rand() & 3) + 1) * ((rand() & 3) + 1);
        if (primeCountHere < combinationCardinality) {
            i--;
            continue;
        }
        int multiplier = rand() | 1;
        multiplierGroups[i] = multiplierGroup { primesForMultiplierGroups.size(), cumulativePairCount, multiplier, primeCountHere };
        for (int iPrime = 0; iPrime < primeCountHere; iPrime++) {
            int quotePrimeUnquote = rand() | 1;
            primesForMultiplierGroups.push_back(quotePrimeUnquote);
        }
        int pairCountHere = primeCountHere * (primeCountHere - 1) / 2;
        cumulativePairCount += pairCountHere;
    }
    
    /*for (int i = 0; i < multiplierGroups.size(); i++) {
        std::cout << "[" << i << "]" << multiplierGroups[i].scoresIndexBase << std::endl;
    }
    std::cout << "[End] " << std::to_string(cumulativePairCount) << std::endl;*/

    int blockCount = 0;
    int threadsPerBlock = 0;
    cudaOccupancyMaxPotentialBlockSize(&blockCount, &threadsPerBlock, rankPrimePairsKernel, 0, 0);
    int warpsPerBlock = threadsPerBlock / WARP_SIZE;
    size_t sharedMemorySizePerBlock = (size_t)rankPrimePairs_sharedMemoryPointers::allocate((uint8_t*)0, warpsPerBlock).endPtr;
    int threadCount = threadsPerBlock * blockCount;
        
    uint* d_primesForMultiplierGroups;
    cudaMalloc((void**)&d_primesForMultiplierGroups, primesForMultiplierGroups.size() * sizeof(uint));
    cudaMemcpy(d_primesForMultiplierGroups, primesForMultiplierGroups.data(), primesForMultiplierGroups.size() * sizeof(uint), cudaMemcpyHostToDevice);
    
    multiplierGroup* d_multiplierGroups;
    cudaMalloc((void**)&d_multiplierGroups, multiplierGroups.size() * sizeof(multiplierGroup));
    cudaMemcpy(d_multiplierGroups, multiplierGroups.data(), multiplierGroups.size() * sizeof(multiplierGroup), cudaMemcpyHostToDevice);
    
    scores.resize(cumulativePairCount);
    scores2.resize(cumulativePairCount);
    float* d_scores;
    cudaMalloc((void**)&d_scores, std::min(cumulativePairCount, (size_t)pairScoreBufferSize) * sizeof(float));
    
    std::vector<batchStartIndex> threadStartIndices(threadCount);
    batchStartIndex* d_threadStartIndices;
    cudaMalloc((void**)&d_threadStartIndices, threadStartIndices.size() * sizeof(batchStartIndex));
    
    {
        runRankPrimePairsKernel<true>(pairScoreBufferSize, cumulativePairCount, threadStartIndices, d_threadStartIndices, multiplierGroups,
            d_multiplierGroups, d_primesForMultiplierGroups, scores, d_scores, blockCount, warpsPerBlock, sharedMemorySizePerBlock);
        int scoresSetCorrectly = 0;
        for (int i = 0; i < cumulativePairCount; i++) {
            if (scores[i] > 0) scoresSetCorrectly++;
        }
        std::cout << testPrefix << "Scores set correctly: " << scoresSetCorrectly << " / " << cumulativePairCount << std::endl;
    }
    
    {
        runRankPrimePairsKernel<true>(pairScoreBufferSize / 4, cumulativePairCount, threadStartIndices, d_threadStartIndices, multiplierGroups,
            d_multiplierGroups, d_primesForMultiplierGroups, scores2, d_scores, blockCount, warpsPerBlock, sharedMemorySizePerBlock);
        int scoresUnchanged = 0;
        for (int i = 0; i < cumulativePairCount; i++) {
            if (scores[i] == scores2[i]) scoresUnchanged++;
        }
        std::cout << testPrefix << "Scores correctly unchanged with different buffer size: " << scoresUnchanged << " / " << cumulativePairCount << std::endl;
    }
    
    {
        int multiplierGroupIndexToChange = rand() % multiplierGroupCount;
        multiplierGroup multiplierGroupToChange = multiplierGroups[multiplierGroupIndexToChange];
        for (int i = 0; i < multiplierGroupToChange.primeCount; i++) {
            primesForMultiplierGroups[multiplierGroupToChange.primesIndexBase + i] ^= 2 << (rand() % 31);
        }
        cudaMemcpy(d_primesForMultiplierGroups, primesForMultiplierGroups.data(), primesForMultiplierGroups.size() * sizeof(uint), cudaMemcpyHostToDevice);
        runRankPrimePairsKernel<true>(pairScoreBufferSize, cumulativePairCount, threadStartIndices, d_threadStartIndices, multiplierGroups,
            d_multiplierGroups, d_primesForMultiplierGroups, scores, d_scores, blockCount, warpsPerBlock, sharedMemorySizePerBlock);
        int scoresChangedCorrectly = 0;
        int scoresUnchangedCorrectly = 0;
        int multiplierGroupToChangePairCount = multiplierGroupToChange.primeCount * (multiplierGroupToChange.primeCount - 1) / 2;
        for (int i = 0; i < cumulativePairCount; i++) {
            bool scoreChanged = (scores[i] != scores2[i]);
            bool shouldBeChanged = (i >= multiplierGroupToChange.scoresIndexBase && i < multiplierGroupToChange.scoresIndexBase + multiplierGroupToChangePairCount);
            if (scoreChanged == shouldBeChanged) {
                if (shouldBeChanged) scoresChangedCorrectly++;
                else scoresUnchangedCorrectly++;
            }
        }
        std::cout << testPrefix << "Scores correctly changed with change to primes in group: " << scoresChangedCorrectly << " / " << multiplierGroupToChangePairCount << std::endl;
        std::cout << testPrefix << "Scores correctly left unchanged with change to primes in group: " << scoresUnchangedCorrectly << " / " << (cumulativePairCount - multiplierGroupToChangePairCount) << std::endl;
    }
    
    {
        int multiplierGroupIndexToChange = rand() % multiplierGroupCount;
        multiplierGroup& multiplierGroupToChange = multiplierGroups[multiplierGroupIndexToChange];
        multiplierGroupToChange.multiplier ^= 2 << (rand() % 31);
        cudaMemcpy(d_multiplierGroups, multiplierGroups.data(), multiplierGroups.size() * sizeof(multiplierGroup), cudaMemcpyHostToDevice);
        runRankPrimePairsKernel<true>(pairScoreBufferSize, cumulativePairCount, threadStartIndices, d_threadStartIndices, multiplierGroups,
            d_multiplierGroups, d_primesForMultiplierGroups, scores2, d_scores, blockCount, warpsPerBlock, sharedMemorySizePerBlock);
        int scoresChangedCorrectly = 0;
        int scoresUnchangedCorrectly = 0;
        int multiplierGroupToChangePairCount = multiplierGroupToChange.primeCount * (multiplierGroupToChange.primeCount - 1) / 2;
        for (int i = 0; i < cumulativePairCount; i++) {
            bool scoreChanged = (scores[i] != scores2[i]);
            bool shouldBeChanged = (i >= multiplierGroupToChange.scoresIndexBase && i < multiplierGroupToChange.scoresIndexBase + multiplierGroupToChangePairCount);
            if (scoreChanged == shouldBeChanged) {
                if (shouldBeChanged) scoresChangedCorrectly++;
                else scoresUnchangedCorrectly++;
            }
        }
        std::cout << testPrefix << "Scores correctly changed with change to group multiplier: " << scoresChangedCorrectly << " / " << multiplierGroupToChangePairCount << std::endl;
        std::cout << testPrefix << "Scores correctly left unchanged with change to group multiplier: " << scoresUnchangedCorrectly << " / " << (cumulativePairCount - multiplierGroupToChangePairCount) << std::endl;
    }
    
    {
        int multiplierGroupIndexToChange = rand() % multiplierGroupCount;
        multiplierGroup multiplierGroupToChange = multiplierGroups[multiplierGroupIndexToChange];
        int multiplierGroupPrimeIndexToChange = rand() % multiplierGroupToChange.primeCount;
        primesForMultiplierGroups[multiplierGroupToChange.primesIndexBase + multiplierGroupPrimeIndexToChange] ^= 2 << (rand() % 31);
        cudaMemcpy(d_primesForMultiplierGroups, primesForMultiplierGroups.data(), primesForMultiplierGroups.size() * sizeof(uint), cudaMemcpyHostToDevice);
        runRankPrimePairsKernel<true>(pairScoreBufferSize, cumulativePairCount, threadStartIndices, d_threadStartIndices, multiplierGroups,
            d_multiplierGroups, d_primesForMultiplierGroups, scores, d_scores, blockCount, warpsPerBlock, sharedMemorySizePerBlock);
        int scoresChangedCorrectly = 0;
        int scoresUnchangedCorrectly = 0;
        int multiplierGroupToChangePairCount = multiplierGroupToChange.primeCount * (multiplierGroupToChange.primeCount - 1) / 2;
        for (int i = 0; i < cumulativePairCount; i++) {
            bool scoreChanged = (scores[i] != scores2[i]);
            bool shouldBeChanged = (i >= multiplierGroupToChange.scoresIndexBase && i < multiplierGroupToChange.scoresIndexBase + multiplierGroupToChangePairCount);
            if (shouldBeChanged) {
                auto primeIndices = extractIndexPair(i - multiplierGroupToChange.scoresIndexBase);
                shouldBeChanged = (multiplierGroupPrimeIndexToChange == primeIndices.a || multiplierGroupPrimeIndexToChange == primeIndices.b);
            }
            if (scoreChanged == shouldBeChanged) {
                if (shouldBeChanged) scoresChangedCorrectly++;
                else scoresUnchangedCorrectly++;
            }
        }
        int pairChangeCountExpected = multiplierGroupToChange.primeCount - 1;
        std::cout << testPrefix << "Scores correctly changed with change to single prime in group: " << scoresChangedCorrectly << " / " << pairChangeCountExpected << std::endl;
        std::cout << testPrefix << "Scores correctly left unchanged with change to single prime in group: " << scoresUnchangedCorrectly << " / " << (cumulativePairCount - pairChangeCountExpected) << std::endl;
    }

    cudaFree(d_scores);
    cudaFree(d_multiplierGroups);
    cudaFree(d_primesForMultiplierGroups);
    cudaFree(d_threadStartIndices);
}
    
void test() {
    testIndicesExhaustive();
    testHeap();
    testFourierScorer();
    testRankPrimePairsKernel();
}

#endif