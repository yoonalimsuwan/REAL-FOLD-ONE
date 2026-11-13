import numpy as np
import matplotlib.pyplot as plt

def generate_doubly_exponential_signal(max_k=4):
    """Generates Paper 6's doubly-exponential block process."""
    signal = []
    true_taus = [0]
    for k in range(max_k + 1):
        tau_k = 2**(2**k) if k > 0 else 1
        block_len = tau_k - true_taus[-1]
        mark = (-1)**k
        signal.extend([mark] * block_len)
        true_taus.append(tau_k)
    return np.array(signal), true_taus

def cusum_detector(signal, threshold, drift=0.1):
    """Page's CUSUM for detecting mean shifts in block processes."""
    s_pos, s_neg = 0.0, 0.0
    change_points = []
    for i, x in enumerate(signal):
        s_pos = max(0, s_pos + x - drift)
        s_neg = max(0, s_neg - x - drift)
        
        if s_pos > threshold or s_neg > threshold:
            change_points.append(i)
            s_pos, s_neg = 0.0, 0.0 # Reset
    return change_points

# --- Execution ---
signal, true_taus = generate_doubly_exponential_signal(max_k=4)
# Threshold scaled to block magnitude
estimated_taus = cusum_detector(signal, threshold=5.0, drift=0.2) 

# Calculate block lengths
true_lengths = np.diff(true_taus)
est_lengths = np.diff([0] + estimated_taus)

print("\n--- Paper 6 CUSUM Change-Point Detection ---")
print(f"True Block Lengths:      {true_lengths}")
print(f"Estimated Block Lengths: {est_lengths[:len(true_lengths)]}")

# Verify Test 2.5 Signature: log(log(ell_k)) grows linearly with k
k_indices = np.arange(1, len(true_lengths))
log_log_true = np.log2(np.log2(true_lengths[1:]))
print(f"log2(log2(ell_k)) slope: {np.polyfit(k_indices, log_log_true, 1)[0]:.2f} (Expected ~1.0)")
