#pragma once
#include "types.hpp"
#include <random>
#include <string>

namespace enn {

// Synthetic data generators for testing and demos
class DataGenerator {
public:
    explicit DataGenerator(unsigned seed = 123) : gen_(seed) {}
    
    // Double-well potential committor data
    // Committor = P(hit x>b before x<a | start at (x,y))
    Batch generate_double_well_committor(int n_samples, F a = -1.0, F b = 1.0,
                                        F x_range = 4.0, F y_range = 2.0);
    
    // Synthetic sequence data for BPTT testing
    SeqBatch generate_copy_task(int batch_size, int seq_len, int vocab_size);
    SeqBatch generate_parity_task(int batch_size, int seq_len);
    SeqBatch generate_adding_task(int batch_size, int seq_len);
    
    // Ornstein-Uhlenbeck process data
    SeqBatch generate_ou_process(int batch_size, int seq_len, F theta = 1.0, 
                                F mu = 0.0, F sigma = 1.0, F dt = 0.01);
    
private:
    std::mt19937 gen_;
    std::uniform_real_distribution<F> uniform_{0.0, 1.0};
    std::normal_distribution<F> normal_{0.0, 1.0};
};

// Simple data loading utilities
class DataLoader {
public:
    // Load CSV data (assumes header with columns: x1, x2, ..., target)
    static Batch load_csv(const std::string& filename);
    
    // Save batch to CSV
    static void save_csv(const Batch& batch, const std::string& filename);
    
    // Load sequence data from CSV (columns: seq_id, step, x1, x2, ..., target)  
    static SeqBatch load_sequence_csv(const std::string& filename);
    
    // Save sequence data to CSV
    static void save_sequence_csv(const SeqBatch& batch, const std::string& filename);
};

// Training utilities
class BatchSampler {
public:
    explicit BatchSampler(unsigned seed = 456) : gen_(seed) {}
    
    // Randomly sample mini-batches from data
    std::vector<Batch> create_batches(const Batch& data, int batch_size, bool shuffle = true);
    
    // Sample mini-batches from sequence data
    std::vector<SeqBatch> create_sequence_batches(const SeqBatch& data, int batch_size, bool shuffle = true);
    
private:
    std::mt19937 gen_;
};

// Metrics and evaluation
struct Metrics {
    F loss;
    F accuracy;      // for classification tasks
    F mse;          // for regression tasks  
    F mae;          // mean absolute error
    int n_samples;
    
    Metrics() : loss(0), accuracy(0), mse(0), mae(0), n_samples(0) {}
    
    void update(F pred, F target, F loss_val) {
        loss += loss_val;
        mse += (pred - target) * (pred - target);
        mae += std::abs(pred - target);
        n_samples++;
        
        // Binary accuracy (threshold at 0.5)
        bool pred_class = pred > 0.5;
        bool true_class = target > 0.5;
        if (pred_class == true_class) accuracy += 1.0;
    }
    
    void finalize() {
        if (n_samples > 0) {
            loss /= n_samples;
            accuracy /= n_samples;
            mse /= n_samples;
            mae /= n_samples;
        }
    }
    
    void reset() {
        loss = accuracy = mse = mae = 0;
        n_samples = 0;
    }
};

} // namespace enn