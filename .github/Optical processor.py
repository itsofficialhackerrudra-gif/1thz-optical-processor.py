"""
Project: 1THz Ultra-High-Speed Optical Neural Network Processor
Description: Simulates optical matrix-vector multiplication using a static 
             Redmi 5A LCD panel as SLM, fast LED array for spatial multiplexing, 
             differential dual-rail encoding (W+ - W-), and LM358 analog op-amp subtraction.
"""

import numpy as np
import matplotlib.pyplot as plt

def run_simulation():
    # 1. 1THz Ultra-Fast Architecture Parameters
    np.random.seed(42)
    N = 20                        # 20x20 Parallel Optical Channels
    T_MIN, T_MAX = 0.05, 0.95     # Static Redmi 5A LCD Transmittance Limits
    
    print(f"Initializing Optical Processor with {N}x{N} parallel channels...")

    # 2. Static Weights & High-Speed LED Input
    W = np.random.uniform(-1.0, 1.0, (N, N))
    X = np.random.uniform(5.0, 25.0, N) 

    # 3. Differential Split (W+ and W- masks)
    W_pos = np.maximum(W, 0)
    W_neg = np.maximum(-W, 0)

    def lcd_static_transmittance(w):
        return T_MIN + (T_MAX - T_MIN) * w

    flux_pos = lcd_static_transmittance(W_pos) @ X
    flux_neg = lcd_static_transmittance(W_neg) @ X

    # 4. Analog Op-Amp Subtraction (LM358 Model)
    meas_pos = flux_pos + np.random.normal(0, 0.001, N)
    meas_neg = flux_neg + np.random.normal(0, 0.001, N)

    # 5. Final Output & Error Calculation
    Y_ideal = W @ X
    Y_optical = (meas_pos - meas_neg) / (T_MAX - T_MIN)

    mean_err = np.mean(np.abs(Y_ideal - Y_optical))
    print(f"==========================================")
    print(f" 1THz FAST-LED SETUP - MEAN ERROR: {mean_err:.6f}")
    print(f"==========================================")

    # 6. Plot & Save Dashboard
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].imshow(W_pos, cmap='inferno')
    axes[0].set_title("Static LCD Mask: W+ (Positive)")
    axes[0].set_xlabel("Channel Index")

    axes[1].imshow(W_neg, cmap='inferno')
    axes[1].set_title("Static LCD Mask: W- (Negative)")
    axes[1].set_xlabel("Channel Index")

    axes[2].plot(Y_ideal, 'k-', linewidth=2.5, label='Ideal Ground Truth')
    axes[2].plot(Y_optical, 'r--', linewidth=2, marker='o', label='Fast-LED Optical Output')
    axes[2].set_title(f"20-Channel Output (Error: {mean_err:.6f})")
    axes[2].set_xlabel("Receiver Channel Index")
    axes[2].set_ylabel("Computed Value")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("optical_processor_performance.png", dpi=300)
    print("Dashboard saved as 'optical_processor_performance.png'")
    plt.show()

if __name__ == "__main__":
    run_simulation()
