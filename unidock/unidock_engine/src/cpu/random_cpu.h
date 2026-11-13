//
// CPU Random Number Generator - Replacement for curand
// Created for Uni-Dock2 CPU backend
//

#ifndef RANDOM_CPU_H
#define RANDOM_CPU_H

#include <random>
#include <cmath>

#ifndef Real
#define Real float
#endif

/**
 * @brief CPU random number generator compatible with CUDA's curand interface
 * Uses Mersenne Twister (mt19937) which provides high-quality randomness
 */
class CPURandom {
private:
    std::mt19937 gen_;
    std::uniform_real_distribution<Real> dist_uniform_{0.0, 1.0};
    std::normal_distribution<Real> dist_normal_{0.0, 1.0};

public:
    /**
     * @brief Initialize random generator with seed
     * @param seed Random seed (similar to curand_init)
     */
    explicit CPURandom(unsigned int seed = 12345) : gen_(seed) {}

    /**
     * @brief Re-seed the generator
     */
    void seed(unsigned int s) {
        gen_.seed(s);
    }

    /**
     * @brief Generate uniform random number in [0, 1)
     * Replacement for curand_uniform()
     */
    Real uniform() {
        return dist_uniform_(gen_);
    }

    /**
     * @brief Generate 4 uniform random numbers in [0, 1)
     * Replacement for curand_uniform4()
     */
    void uniform4(Real* out) {
        out[0] = uniform();
        out[1] = uniform();
        out[2] = uniform();
        out[3] = uniform();
    }

    /**
     * @brief Generate normal/gaussian random number with mean=0, std=1
     */
    Real normal() {
        return dist_normal_(gen_);
    }

    /**
     * @brief Generate random integer in range [min, max]
     * Replacement for curand() % range
     */
    int randint(int min, int max) {
        std::uniform_int_distribution<int> dist(min, max);
        return dist(gen_);
    }

    /**
     * @brief Generate random real in range [min, max]
     */
    Real rand_real_within(Real min, Real max) {
        return min + uniform() * (max - min);
    }

    /**
     * @brief Generate 4 random numbers forming a point in unit sphere
     * Used for random orientation generation
     */
    void rand_in_sphere(Real* out_rand_4) {
        while (true) {
            out_rand_4[0] = (uniform() - 0.5) * 2.0;  // [-1, 1]
            out_rand_4[1] = (uniform() - 0.5) * 2.0;
            out_rand_4[2] = (uniform() - 0.5) * 2.0;
            
            Real norm_sq = out_rand_4[0] * out_rand_4[0] + 
                          out_rand_4[1] * out_rand_4[1] + 
                          out_rand_4[2] * out_rand_4[2];
            
            if (norm_sq < 1.0) {
                out_rand_4[3] = uniform();
                return;
            }
        }
    }
};


/**
 * @brief Thread-local random generator for OpenMP parallelization
 * Each thread gets its own generator to avoid race conditions
 */
class ThreadLocalRandom {
private:
    std::vector<CPURandom> generators_;
    int base_seed_;

public:
    explicit ThreadLocalRandom(int seed, int num_threads) 
        : base_seed_(seed) {
        generators_.reserve(num_threads);
        for (int i = 0; i < num_threads; i++) {
            generators_.emplace_back(seed + i * 1000000);
        }
    }

    CPURandom& get(int thread_id) {
        return generators_[thread_id];
    }

    /**
     * @brief Get generator for current OpenMP thread
     */
    CPURandom& get_current() {
#if USE_OPENMP
        return generators_[omp_get_thread_num()];
#else
        return generators_[0];
#endif
    }
};

#endif // RANDOM_CPU_H
