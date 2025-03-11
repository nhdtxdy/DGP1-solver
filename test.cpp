#include <iostream>
#include <bitset>
#include <boost/dynamic_bitset.hpp>
#include <chrono>
#include <random>

constexpr size_t N = 1000000;  // Number of bits
constexpr int iterations = 1000;  // Number of benchmark iterations

int main() {
    // Setup a random generator to initialize bitsets
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);

    // Initialize std::bitset and boost::dynamic_bitset
    std::bitset<N> bs;
    boost::dynamic_bitset<> dbs(N);

    for (size_t i = 0; i < N; ++i) {
        bool bit = dis(gen);
        bs[i] = bit;
        dbs[i] = bit;
    }

    volatile size_t dummy = 0;  // Volatile to prevent optimizations

    // Benchmark count operation
    auto start_bs_count = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        dummy += bs.count();
    }
    auto end_bs_count = std::chrono::high_resolution_clock::now();
    auto time_bs_count = std::chrono::duration_cast<std::chrono::microseconds>(end_bs_count - start_bs_count).count();

    auto start_dbs_count = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        dummy += dbs.count();
    }
    auto end_dbs_count = std::chrono::high_resolution_clock::now();
    auto time_dbs_count = std::chrono::duration_cast<std::chrono::microseconds>(end_dbs_count - start_dbs_count).count();

    std::cout << "std::bitset count total time: " << time_bs_count << " microseconds\n";
    std::cout << "boost::dynamic_bitset count total time: " << time_dbs_count << " microseconds\n";

    // Benchmark flip (invert all bits)
    auto start_bs_flip = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        bs.flip();
    }
    auto end_bs_flip = std::chrono::high_resolution_clock::now();
    auto time_bs_flip = std::chrono::duration_cast<std::chrono::microseconds>(end_bs_flip - start_bs_flip).count();

    auto start_dbs_flip = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        dbs.flip();
    }
    auto end_dbs_flip = std::chrono::high_resolution_clock::now();
    auto time_dbs_flip = std::chrono::duration_cast<std::chrono::microseconds>(end_dbs_flip - start_dbs_flip).count();

    std::cout << "std::bitset flip total time: " << time_bs_flip << " microseconds\n";
    std::cout << "boost::dynamic_bitset flip total time: " << time_dbs_flip << " microseconds\n";

    // Prepare copies for binary operations
    std::bitset<N> bs2 = bs;
    boost::dynamic_bitset<> dbs2 = dbs;

    // Benchmark bitwise OR
    auto start_bs_or = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = bs | bs2;
        dummy += result.count();
    }
    auto end_bs_or = std::chrono::high_resolution_clock::now();
    auto time_bs_or = std::chrono::duration_cast<std::chrono::microseconds>(end_bs_or - start_bs_or).count();

    auto start_dbs_or = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = dbs | dbs2;
        dummy += result.count();
    }
    auto end_dbs_or = std::chrono::high_resolution_clock::now();
    auto time_dbs_or = std::chrono::duration_cast<std::chrono::microseconds>(end_dbs_or - start_dbs_or).count();

    std::cout << "std::bitset OR total time: " << time_bs_or << " microseconds\n";
    std::cout << "boost::dynamic_bitset OR total time: " << time_dbs_or << " microseconds\n";

    // Benchmark left shift (<< 1)
    auto start_bs_lshift = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = bs << 1;
        dummy += result.count();
    }
    auto end_bs_lshift = std::chrono::high_resolution_clock::now();
    auto time_bs_lshift = std::chrono::duration_cast<std::chrono::microseconds>(end_bs_lshift - start_bs_lshift).count();

    auto start_dbs_lshift = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = dbs << 1;
        dummy += result.count();
    }
    auto end_dbs_lshift = std::chrono::high_resolution_clock::now();
    auto time_dbs_lshift = std::chrono::duration_cast<std::chrono::microseconds>(end_dbs_lshift - start_dbs_lshift).count();

    std::cout << "std::bitset left shift total time: " << time_bs_lshift << " microseconds\n";
    std::cout << "boost::dynamic_bitset left shift total time: " << time_dbs_lshift << " microseconds\n";

    // Benchmark right shift (>> 1)
    auto start_bs_rshift = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = bs >> 1;
        dummy += result.count();
    }
    auto end_bs_rshift = std::chrono::high_resolution_clock::now();
    auto time_bs_rshift = std::chrono::duration_cast<std::chrono::microseconds>(end_bs_rshift - start_bs_rshift).count();

    auto start_dbs_rshift = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = dbs >> 1;
        dummy += result.count();
    }
    auto end_dbs_rshift = std::chrono::high_resolution_clock::now();
    auto time_dbs_rshift = std::chrono::duration_cast<std::chrono::microseconds>(end_dbs_rshift - start_dbs_rshift).count();

    std::cout << "std::bitset right shift total time: " << time_bs_rshift << " microseconds\n";
    std::cout << "boost::dynamic_bitset right shift total time: " << time_dbs_rshift << " microseconds\n";

    // Benchmark bitwise XOR
    auto start_bs_xor = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = bs ^ bs2;
        dummy += result.count();
    }
    auto end_bs_xor = std::chrono::high_resolution_clock::now();
    auto time_bs_xor = std::chrono::duration_cast<std::chrono::microseconds>(end_bs_xor - start_bs_xor).count();

    auto start_dbs_xor = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto result = dbs ^ dbs2;
        dummy += result.count();
    }
    auto end_dbs_xor = std::chrono::high_resolution_clock::now();
    auto time_dbs_xor = std::chrono::duration_cast<std::chrono::microseconds>(end_dbs_xor - start_dbs_xor).count();

    std::cout << "std::bitset XOR total time: " << time_bs_xor << " microseconds\n";
    std::cout << "boost::dynamic_bitset XOR total time: " << time_dbs_xor << " microseconds\n";

    return 0;
}
