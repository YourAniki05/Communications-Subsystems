import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path

def calculate_ber(decoded_bits, meta_data_path):
    """
    Calculates the Bit Error Rate by comparing decoded bits to the ground truth.
    Returns BER and number of errors.
    """
    try:
        # 1. Load the ground truth bits from the metadata
        with open(meta_data_path) as f:
            meta = json.load(f)
        
        # 2. Extract ground truth bits - use 'clean_bits' as found in your metadata
        if 'clean_bits' in meta:
            ground_truth_bits = np.array(meta['clean_bits'])
            print(f"Found ground truth bits under key: 'clean_bits'")
        else:
            print("ERROR: Could not find ground truth bits in metadata. Available keys:", list(meta.keys()))
            return None, None, None
        
        # 3. Ensure lengths match
        min_length = min(len(decoded_bits), len(ground_truth_bits))
        decoded_bits_truncated = decoded_bits[:min_length]
        ground_truth_bits_truncated = ground_truth_bits[:min_length]
        
        # 4. Calculate number of errors and BER
        num_errors = np.sum(decoded_bits_truncated != ground_truth_bits_truncated)
        ber = num_errors / min_length
        
        return ber, num_errors, min_length
        
    except Exception as e:
        print(f"Error calculating BER: {e}")
        return None, None, None

def main():
    # ==========================================================================
    # 1. CONFIGURATION - Using pathlib for better path handling
    # ==========================================================================
    project_root = Path(__file__).parent.parent
    snr_level = 'snr_5db'
    sample_name = 'sample_002'
    samples_per_symbol = 8
    
    # ==========================================================================
    # 2. PATH SETUP - More efficient with pathlib
    # ==========================================================================
    data_dir = project_root / 'data' / 'cubesat_dataset' / 'phase1_timing' / snr_level / sample_name
    rx_file_path = data_dir / 'rx.npy'
    meta_file_path = data_dir / 'meta.json'
    
    # Output directories
    sample_results_dir = project_root / 'results' / 'phase1' / snr_level / sample_name
    sample_results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {rx_file_path}")
    
    if not rx_file_path.exists():
        print(f"ERROR: File not found at {rx_file_path}")
        return

    # ==========================================================================
    # 3. DATA LOADING - Single operation
    # ==========================================================================
    baseband_signal = np.load(rx_file_path)
    with open(meta_file_path) as f:
        metadata = json.load(f)

    print(f"Loaded {len(baseband_signal):,} samples")

    # ==========================================================================
    # 4. EFFICIENT MATCHED FILTERING
    # ==========================================================================
    # Precompute matched filter
    matched_filter = np.ones(samples_per_symbol) / np.sqrt(samples_per_symbol)
    
    # Use faster convolution with FFT (much faster for long signals)
    filtered_signal = signal.fftconvolve(baseband_signal, matched_filter, mode='same')

    # ==========================================================================
    # 5. OPTIMIZED TIMING RECOVERY
    # ==========================================================================
    print("Performing timing recovery...")
    
    # Vectorized energy calculation - much faster than loop
    offsets = np.arange(samples_per_symbol)
    energies = np.array([
        np.mean(np.abs(filtered_signal[offset::samples_per_symbol])**2)
        for offset in offsets
    ])
    
    best_offset = np.argmax(energies)
    max_energy = energies[best_offset]
    
    print(f"Best timing offset: {best_offset} samples (energy: {max_energy:.4f})")
    
    # Efficient slicing with step
    symbols_recovered = filtered_signal[best_offset::samples_per_symbol]

    # ==========================================================================
    # 6. DEMODULATION - Vectorized operation
    # ==========================================================================
    decoded_bits = (np.real(symbols_recovered) > 0).astype(np.int8)  # Use int8 to save memory

    # ==========================================================================
    # 7. BIT ERROR RATE CALCULATION - CRITICAL FOR CHALLENGE
    # ==========================================================================
    print("\n" + "="*50)
    print("BIT ERROR RATE ANALYSIS")
    print("="*50)
    
    ber, num_errors, num_bits_compared = calculate_ber(decoded_bits, meta_file_path)
    
    if ber is not None:
        print(f"Errors: {num_errors} / {num_bits_compared} bits")
        print(f"BER: {ber:.6f} ({ber * 100:.4f}%)")
        print(f"Target: BER ≤ 0.01 (1.00%)")
        
        if ber <= 0.01:
            print("✅ SUCCESS: BER requirement met!")
            ber_status = "PASS"
        else:
            print("❌ FAILED: BER too high.")
            ber_status = "FAIL"
    else:
        print("⚠️  Could not calculate BER")
        ber_status = "ERROR"

    # ==========================================================================
    # 8. SAVE RESULTS
    # ==========================================================================
    bits_output_path = sample_results_dir / 'decoded_bits.npy'
    np.save(bits_output_path, decoded_bits)
    print(f"\nDecoded bits saved to: {bits_output_path}")
    
    # ==========================================================================
    # 9. EFFICIENT PLOTTING - Batch create figures
    # ==========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot 1: Raw Signal
    axes[0].plot(np.real(baseband_signal[:1000]), label='In-Phase', alpha=0.8, linewidth=0.8)
    axes[0].plot(np.imag(baseband_signal[:1000]), label='Quadrature', alpha=0.8, linewidth=0.8)
    axes[0].set_title(f"Raw Signal - {snr_level}/{sample_name}")
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Filtered Signal
    axes[1].plot(np.real(filtered_signal[:500]), label='Filtered I', alpha=0.8, linewidth=0.8)
    axes[1].plot(np.imag(filtered_signal[:500]), label='Filtered Q', alpha=0.8, linewidth=0.8)
    axes[1].set_title("Filtered Signal")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Timing Recovery
    axes[2].plot(offsets, energies, 'bo-', linewidth=2, markersize=6)
    axes[2].axvline(best_offset, color='red', linestyle='--', label=f'Best: {best_offset}')
    axes[2].set_title("Timing Recovery - Energy vs Offset")
    axes[2].set_xlabel("Sampling Offset")
    axes[2].set_ylabel("Symbol Energy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Constellation Diagram
    n_symbols_plot = min(2000, len(symbols_recovered))
    indices = np.random.choice(len(symbols_recovered), n_symbols_plot, replace=False)
    axes[3].scatter(np.real(symbols_recovered[indices]), np.imag(symbols_recovered[indices]),
                   alpha=0.5, s=10, c='blue', edgecolors='none')
    axes[3].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[3].axvline(0, color='black', linestyle='-', alpha=0.3)
    axes[3].set_title(f"Constellation Diagram\n{n_symbols_plot} symbols")
    axes[3].set_xlabel("In-Phase")
    axes[3].set_ylabel("Quadrature")
    axes[3].grid(True, alpha=0.3)
    axes[3].axis('equal')

    # Plot 5: Eye Diagram
    num_eyes = min(30, len(filtered_signal) // samples_per_symbol)
    eye_data = np.zeros((num_eyes, 2 * samples_per_symbol))
    
    for i in range(num_eyes):
        start_idx = i * samples_per_symbol
        end_idx = start_idx + 2 * samples_per_symbol
        if end_idx < len(filtered_signal):
            eye_data[i] = np.real(filtered_signal[start_idx:end_idx])
    
    time_axis = np.arange(2 * samples_per_symbol)
    mean_eye = np.mean(eye_data, axis=0)
    std_eye = np.std(eye_data, axis=0)
    
    axes[4].plot(time_axis, mean_eye, 'b-', linewidth=2, label='Mean')
    axes[4].fill_between(time_axis, mean_eye - std_eye, mean_eye + std_eye,
                        alpha=0.3, color='blue', label='±1 STD')
    axes[4].axvline(samples_per_symbol + best_offset, color='red', linestyle='--',
                   label=f'Sampling point')
    axes[4].set_title("Eye Diagram (Real Component)")
    axes[4].set_xlabel("Sample Offset")
    axes[4].set_ylabel("Amplitude")
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    # Plot 6: Bit Stream with error highlighting (if BER was calculated)
    n_bits_plot = min(100, len(decoded_bits))
    axes[5].step(range(n_bits_plot), decoded_bits[:n_bits_plot], where='post', linewidth=1.5, label='Decoded')
    
    if ber is not None and num_bits_compared >= n_bits_plot:
        # Plot ground truth for comparison
        with open(meta_file_path) as f:
            meta = json.load(f)
        if 'clean_bits' in meta:
            ground_truth_bits = np.array(meta['clean_bits'])[:n_bits_plot]
        
        if ground_truth_bits is not None:
            axes[5].step(range(n_bits_plot), ground_truth_bits, where='post', linewidth=1.5, 
                        alpha=0.7, linestyle='--', label='Ground Truth', color='red')
            axes[5].legend()
    
    axes[5].set_title(f"Decoded Bits (First {n_bits_plot} bits)\nBER: {ber:.4f}" if ber is not None else f"Decoded Bits (First {n_bits_plot} bits)")
    axes[5].set_xlabel("Bit Index")
    axes[5].set_ylabel("Bit Value")
    axes[5].set_yticks([0, 1])
    axes[5].grid(True, alpha=0.3)

    # Save all plots
    plt.tight_layout()
    composite_path = sample_results_dir / 'composite_analysis.png'
    plt.savefig(composite_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Composite analysis plot saved to: {composite_path}")

    # ==========================================================================
    # 10. SAVE PROCESSING SUMMARY WITH BER RESULTS
    # ==========================================================================
    summary = {
        'metadata': metadata,
        'processing': {
            'snr_level': snr_level,
            'sample_name': sample_name,
            'samples_loaded': len(baseband_signal),
            'symbols_recovered': len(symbols_recovered),
            'bits_decoded': len(decoded_bits),
            'timing_offset': int(best_offset),
            'samples_per_symbol': samples_per_symbol,
            'max_energy': float(max_energy),
        },
        'ber_analysis': {
            'status': ber_status,
            'ber': float(ber) if ber is not None else None,
            'num_errors': int(num_errors) if num_errors is not None else None,
            'num_bits_compared': int(num_bits_compared) if num_bits_compared is not None else None,
            'target_ber': 0.01
        }
    }
    
    summary_path = sample_results_dir / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Processing summary saved to: {summary_path}")

    # ==========================================================================
    # 11. FINAL REPORT
    # ==========================================================================
    print("\n" + "="*60)
    print("PHASE 1 PROCESSING COMPLETE")
    print("="*60)
    print(f"Processed: {snr_level}/{sample_name}")
    print(f"Input: {len(baseband_signal):,} samples")
    print(f"Output: {len(decoded_bits):,} bits")
    
    if ber is not None:
        print(f"BER: {ber:.6f} ({num_errors} errors in {num_bits_compared} bits)")
        print(f"Status: {ber_status} (Target: BER ≤ 0.01)")
    
    print(f"Results saved to: {sample_results_dir}")
    print("="*60)

if __name__ == "__main__":
    main()