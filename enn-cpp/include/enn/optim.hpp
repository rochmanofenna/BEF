#pragma once
#include "types.hpp"
#include <cmath>

namespace enn {

// Adam optimizer with bias correction
struct Adam {
    F lr;      // learning rate
    F beta1;   // exponential decay rate for first moment
    F beta2;   // exponential decay rate for second moment  
    F eps;     // small constant for numerical stability
    int t;     // time step counter
    
    explicit Adam(F lr_ = 1e-3, F beta1_ = 0.9, F beta2_ = 0.999, F eps_ = 1e-8)
        : lr(lr_), beta1(beta1_), beta2(beta2_), eps(eps_), t(0) {}
    
    // Update parameter with gradient (matrix version)
    void step(Mat& param, Mat& m, Mat& v, const Mat& grad) {
        t++;
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);
        
        Mat m_hat = m / (1.0 - std::pow(beta1, t));
        Mat v_hat = v / (1.0 - std::pow(beta2, t));
        
        param -= lr * m_hat.cwiseQuotient((v_hat.array().sqrt() + eps).matrix());
    }
    
    // Update parameter with gradient (vector version)
    void step(Vec& param, Vec& m, Vec& v, const Vec& grad) {
        t++;
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad.cwiseProduct(grad);
        
        Vec m_hat = m / (1.0 - std::pow(beta1, t));
        Vec v_hat = v / (1.0 - std::pow(beta2, t));
        
        param -= lr * m_hat.cwiseQuotient((v_hat.array().sqrt() + eps).matrix());
    }
    
    // Update parameter with gradient (scalar version)
    void step(F& param, F& m, F& v, F grad) {
        t++;
        m = beta1 * m + (1.0 - beta1) * grad;
        v = beta2 * v + (1.0 - beta2) * grad * grad;
        
        F m_hat = m / (1.0 - std::pow(beta1, t));
        F v_hat = v / (1.0 - std::pow(beta2, t));
        
        param -= lr * m_hat / (std::sqrt(v_hat) + eps);
    }
    
    void reset() { t = 0; }
};

// AdamW optimizer (Adam with decoupled weight decay)
struct AdamW : Adam {
    F weight_decay;
    
    explicit AdamW(F lr_ = 1e-3, F beta1_ = 0.9, F beta2_ = 0.999, 
                   F eps_ = 1e-8, F weight_decay_ = 1e-4)
        : Adam(lr_, beta1_, beta2_, eps_), weight_decay(weight_decay_) {}
    
    void step(Mat& param, Mat& m, Mat& v, const Mat& grad) {
        // Weight decay applied directly to parameters
        param *= (1.0 - lr * weight_decay);
        Adam::step(param, m, v, grad);
    }
    
    void step(Vec& param, Vec& m, Vec& v, const Vec& grad) {
        param *= (1.0 - lr * weight_decay);
        Adam::step(param, m, v, grad);
    }
    
    void step(F& param, F& m, F& v, F grad) {
        param *= (1.0 - lr * weight_decay);
        Adam::step(param, m, v, grad);
    }
};

// Learning rate schedulers
struct CosineScheduler {
    F base_lr;     // initial learning rate
    F min_lr;      // minimum learning rate
    int T_max;     // maximum number of iterations
    
    CosineScheduler(F base_lr_, F min_lr_, int T_max_) 
        : base_lr(base_lr_), min_lr(min_lr_), T_max(T_max_) {}
    
    F operator()(int t) const {
        if (t >= T_max) return min_lr;
        return min_lr + (base_lr - min_lr) * (1.0 + std::cos(M_PI * t / T_max)) / 2.0;
    }
};

struct LinearScheduler {
    F base_lr;
    F final_lr;
    int T_max;
    
    LinearScheduler(F base_lr_, F final_lr_, int T_max_)
        : base_lr(base_lr_), final_lr(final_lr_), T_max(T_max_) {}
        
    F operator()(int t) const {
        if (t >= T_max) return final_lr;
        F alpha = static_cast<F>(t) / T_max;
        return base_lr + alpha * (final_lr - base_lr);
    }
};

} // namespace enn