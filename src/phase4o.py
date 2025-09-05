import numpy as np
import json
import os
import matplotlib.pyplot as plt

print("=" * 70)
print("PHASE 4: DOPPLER SHIFT PROCESSING")
print("=" * 70)

# Configuration - Let's verify these paths exist
BASE_DATA_PATH = r"C:\Communications Test\Communications-Subsystem\data\cubesat_dataset\phase4_doppler"
RESULTS_DIR = r"C:\Communications Test\Communications-Subsystem\results\phase4"

def create_directories():
    """Create necessary directories"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"✓ Results directory ready: {RESULTS_DIR}")
    print(f"✓ Results will be saved to: {RESULTS_DIR}")

def find_doppler_samples():
    """Find all Doppler samples based on the exact structure"""
    sample_paths = []
    
    print(f"Looking for data in: {BASE_DATA_PATH}")
    
    if not os.path.exists(BASE_DATA_PATH):
        print(f"❌ CRITICAL ERROR: Phase 4 data path not found: {BASE_DATA_PATH}")
        print("Please check if:")
        print("1. The path is correct")
        print("2. The data files are downloaded")
        print("3. The folder structure matches expected format")
        return []
    
    print("✓ Data directory found. Scanning structure...")
    
    # List all items in the base directory
    items = os.listdir(BASE_DATA_PATH)
    print(f"Items in base directory: {items}")
    
    # Look for Doppler folders (doppler_-5000Hz, doppler_2000Hz, etc.)
    for item in items:
        item_path = os.path.join(BASE_DATA_PATH, item)
        
        if os.path.isdir(item_path) and 'doppler' in item.lower():
            print(f"Found Doppler folder: {item}")
            
            # Look for SNR folders inside each Doppler folder
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                
                if os.path.isdir(sub_item_path) and 'snr' in sub_item.lower():
                    print(f"  Found SNR folder: {sub_item}")
                    
                    # Check if rx.npy exists in this SNR directory
                    rx_file = os.path.join(sub_item_path, "rx.npy")
                    meta_file = os.path.join(sub_item_path, "meta.json")
                    
                    if os.path.exists(rx_file):
                        print(f"    ✓ Found rx.npy")
                    else:
                        print(f"    ❌ Missing rx.npy")
                        
                    if os.path.exists(meta_file):
                        print(f"    ✓ Found meta.json")
                    else:
                        print(f"    ❌ Missing meta.json")
                    
                    if os.path.exists(rx_file) and os.path.exists(meta_file):
                        sample_paths.append(sub_item_path)
                        print(f"    ✓ Added to processing queue")
                    else:
                        print(f"    ⚠️  Skipping - missing required files")
    
    print(f"\nTotal Doppler samples found: {len(sample_paths)}")
    return sample_paths

def extract_doppler_from_path(sample_path):
    """Extract Doppler value from the folder path"""
    doppler_folder = os.path.basename(os.path.dirname(sample_path))
    
    # Extract Doppler value from folder name (e.g., "doppler_-5000Hz" -> -5000)
    if 'doppler' in doppler_folder.lower():
        # Remove 'doppler_' prefix and 'Hz' suffix
        doppler_str = doppler_folder.replace('doppler_', '').replace('Hz', '')
        try:
            return int(doppler_str)
        except ValueError:
            print(f"Warning: Could not parse Doppler value from {doppler_folder}")
            return 0
    return 0

def estimate_frequency_offset(signal, sample_rate=32000):
    """Estimate frequency offset using FFT peak detection"""
    try:
        # Remove DC offset
        signal = signal - np.mean(signal)
        
        # Compute FFT
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), 1/sample_rate)
        
        # Find peak frequency (excluding DC)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_result[:len(fft_result)//2])
        positive_fft[0] = 0  # Remove DC component
        
        peak_freq = positive_freqs[np.argmax(positive_fft)]
        return peak_freq
    except Exception as e:
        print(f"Error in frequency estimation: {e}")
        return 0

def correct_frequency_offset(signal, freq_offset, sample_rate=32000):
    """Correct frequency offset using complex exponential"""
    try:
        t = np.arange(len(signal)) / sample_rate
        correction = np.exp(-1j * 2 * np.pi * freq_offset * t)
        return signal * correction
    except Exception as e:
        print(f"Error in frequency correction: {e}")
        return signal

def load_metadata(sample_path):
    """Load metadata from file"""
    meta_file = os.path.join(sample_path, "meta.json")
    try:
        with open(meta_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}

def get_sample_id(sample_path):
    """Create a unique identifier for the sample"""
    doppler_folder = os.path.basename(os.path.dirname(sample_path))
    snr_folder = os.path.basename(sample_path)
    return f"{doppler_folder}_{snr_folder}"

def process_doppler_sample(sample_path):
    """Process a single Doppler sample"""
    try:
        doppler_value = extract_doppler_from_path(sample_path)
        sample_id = get_sample_id(sample_path)
        
        print(f"\n📡 Processing: {sample_id}")
        print(f"   Doppler: {doppler_value} Hz")
        print(f"   Data path: {sample_path}")
        
        # Load files
        rx_file = os.path.join(sample_path, "rx.npy")
        meta_file = os.path.join(sample_path, "meta.json")
        
        # Check if files exist
        if not os.path.exists(rx_file):
            print(f"   ❌ rx.npy not found at: {rx_file}")
            return None
        if not os.path.exists(meta_file):
            print(f"   ❌ meta.json not found at: {meta_file}")
            return None
            
        signal_data = np.load(rx_file)
        metadata = load_metadata(sample_path)
        
        print(f"   Signal length: {len(signal_data)} samples")
        
        # Get parameters
        sample_rate = metadata.get('sample_rate', 32000)
        sps = metadata.get('sps', 8)
        snr = metadata.get('snr_db', 10)
        
        print(f"   Sample rate: {sample_rate} Hz")
        print(f"   SNR: {snr} dB")
        print(f"   SPS: {sps}")
        
        # Estimate frequency offset
        estimated_offset = estimate_frequency_offset(signal_data, sample_rate)
        print(f"   Estimated offset: {estimated_offset:.2f} Hz")
        print(f"   Estimation error: {abs(estimated_offset - doppler_value):.2f} Hz")
        
        # Correct frequency offset
        corrected_signal = correct_frequency_offset(signal_data, estimated_offset, sample_rate)
        
        # Demodulate both original and corrected signals
        symbols_original = signal_data[::sps]
        symbols_corrected = corrected_signal[::sps]
        
        bits_original = (np.real(symbols_original) > 0).astype(int)
        bits_corrected = (np.real(symbols_corrected) > 0).astype(int)
        
        # Calculate BER if ground truth is available
        ber_original = None
        ber_corrected = None
        
        if 'clean_bits' in metadata:
            true_bits = np.array(metadata['clean_bits'])
            min_len = min(len(bits_original), len(true_bits))
            
            errors_original = np.sum(bits_original[:min_len] != true_bits[:min_len])
            errors_corrected = np.sum(bits_corrected[:min_len] != true_bits[:min_len])
            
            ber_original = errors_original / min_len if min_len > 0 else 0
            ber_corrected = errors_corrected / min_len if min_len > 0 else 0
            
            print(f"   BER Original: {ber_original:.6f}")
            print(f"   BER Corrected: {ber_corrected:.6f}")
            
            if ber_original > 0:
                improvement = ((ber_original - ber_corrected) / ber_original * 100)
                print(f"   Improvement: {improvement:.1f}%")
        else:
            print("   ⚠️  No ground truth bits found in metadata")
        
        # Save results to RESULTS_DIR
        sample_results_dir = os.path.join(RESULTS_DIR, sample_id)
        os.makedirs(sample_results_dir, exist_ok=True)
        
        np.save(os.path.join(sample_results_dir, "corrected_signal.npy"), corrected_signal)
        np.save(os.path.join(sample_results_dir, "decoded_bits.npy"), bits_corrected)
        
        print(f"   ✓ Saved results to: {sample_results_dir}")
        
        # Create analysis plots in RESULTS_DIR
        create_doppler_plots(sample_results_dir, sample_id, signal_data, corrected_signal, 
                           symbols_original, symbols_corrected, estimated_offset, 
                           doppler_value, ber_original, ber_corrected, metadata)
        
        return {
            'sample_id': sample_id,
            'path': sample_path,
            'doppler_hz': doppler_value,
            'estimated_offset': float(estimated_offset),
            'snr': snr,
            'ber_original': float(ber_original) if ber_original is not None else None,
            'ber_corrected': float(ber_corrected) if ber_corrected is not None else None,
            'error_hz': float(abs(estimated_offset - doppler_value))
        }
        
    except Exception as e:
        print(f"   ❌ Error processing sample: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_doppler_plots(results_dir, sample_id, original_signal, corrected_signal, symbols_orig, 
                        symbols_corr, estimated_offset, true_doppler, ber_orig, ber_corr, metadata):
    """Create comprehensive Doppler analysis plots"""
    try:
        plt.figure(figsize=(15, 10))
        
        # Time domain signals
        plt.subplot(2, 3, 1)
        time_pts = min(500, len(original_signal))
        plt.plot(np.real(original_signal[:time_pts]), 'b-', label='Original', linewidth=1)
        plt.plot(np.real(corrected_signal[:time_pts]), 'r-', label='Corrected', linewidth=1, alpha=0.7)
        plt.title('Time Domain: Original vs Corrected', fontweight='bold')
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Frequency spectrum - Original
        plt.subplot(2, 3, 2)
        fft_orig = np.fft.fft(original_signal)
        freqs = np.fft.fftfreq(len(original_signal), 1/32000)
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = np.abs(fft_orig[:len(fft_orig)//2])
        plt.plot(positive_freqs, positive_fft, 'b-')
        plt.title('Frequency Spectrum - Original', fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(-2000, 2000)
        
        # Frequency spectrum - Corrected
        plt.subplot(2, 3, 3)
        fft_corr = np.fft.fft(corrected_signal)
        positive_fft_corr = np.abs(fft_corr[:len(fft_corr)//2])
        plt.plot(positive_freqs, positive_fft_corr, 'r-')
        plt.title('Frequency Spectrum - Corrected', fontweight='bold')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True, alpha=0.3)
        plt.xlim(-2000, 2000)
        
        # Constellation - Original
        plt.subplot(2, 3, 4)
        plt.scatter(np.real(symbols_orig[:1000]), np.imag(symbols_orig[:1000]), 
                   alpha=0.5, s=3, c='blue')
        plt.title('Constellation - Original', fontweight='bold')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Constellation - Corrected
        plt.subplot(2, 3, 5)
        plt.scatter(np.real(symbols_corr[:1000]), np.imag(symbols_corr[:1000]), 
                   alpha=0.5, s=3, c='red')
        plt.title('Constellation - Corrected', fontweight='bold')
        plt.xlabel('In-Phase')
        plt.ylabel('Quadrature')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Information panel
        plt.subplot(2, 3, 6)
        error = abs(estimated_offset - true_doppler)
        
        info_text = f"""DOPPLER ANALYSIS:
{sample_id}

True Doppler: {true_doppler} Hz
Estimated: {estimated_offset:.1f} Hz
Error: {error:.1f} Hz

SNR: {metadata.get('snr_db', 'N/A')} dB
Sample Rate: {metadata.get('sample_rate', 32000)} Hz"""

        if ber_orig is not None and ber_corr is not None:
            info_text += f"\n\nBER Original: {ber_orig:.6f}"
            info_text += f"\nBER Corrected: {ber_corr:.6f}"
            if ber_orig > 0:
                improvement = ((ber_orig - ber_corr) / ber_orig * 100)
                info_text += f"\nImprovement: {improvement:.1f}%"
        
        plt.text(0.1, 0.5, info_text, fontfamily='monospace', fontsize=9,
                bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8))
        plt.axis('off')
        plt.title('Performance Summary', fontweight='bold')
        
        plt.tight_layout()
        plot_path = os.path.join(results_dir, "doppler_analysis.png")
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Analysis plot saved to: {plot_path}")
        
    except Exception as e:
        print(f"   ⚠️  Could not create plots: {str(e)}")

def create_summary_plots(results):
    """Create summary performance plots"""
    if not results:
        return
    
    try:
        # Doppler estimation accuracy plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        doppler_values = [r['doppler_hz'] for r in results]
        errors = [r['error_hz'] for r in results]
        snr_values = [r['snr'] for r in results]
        
        # Color points by SNR
        scatter = plt.scatter(doppler_values, errors, c=snr_values, s=100, alpha=0.7, cmap='viridis')
        plt.colorbar(scatter, label='SNR (dB)')
        plt.xlabel('True Doppler Shift (Hz)', fontweight='bold')
        plt.ylabel('Estimation Error (Hz)', fontweight='bold')
        plt.title('Doppler Estimation Accuracy', fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # BER improvement plot
        plt.subplot(1, 2, 2)
        bers_original = [r['ber_original'] for r in results if r['ber_original'] is not None]
        bers_corrected = [r['ber_corrected'] for r in results if r['ber_corrected'] is not None]
        
        if bers_original and bers_corrected:
            x_pos = np.arange(len(bers_original))
            width = 0.35
            
            plt.bar(x_pos - width/2, bers_original, width, label='Before Correction', alpha=0.7)
            plt.bar(x_pos + width/2, bers_corrected, width, label='After Correction', alpha=0.7)
            
            plt.xlabel('Sample Index', fontweight='bold')
            plt.ylabel('Bit Error Rate', fontweight='bold')
            plt.title('BER Before vs After Doppler Correction', fontweight='bold')
            plt.yscale('log')
            plt.legend()
            plt.grid(True, alpha=0.3, axis='y')
        else:
            plt.text(0.5, 0.5, 'BER data not available', ha='center', va='center')
            plt.title('BER Comparison', fontweight='bold')
        
        plt.tight_layout()
        summary_plot = os.path.join(RESULTS_DIR, "doppler_performance_summary.png")
        plt.savefig(summary_plot, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Summary performance plot saved: {summary_plot}")
        
    except Exception as e:
        print(f"⚠️  Could not create summary plots: {str(e)}")

def main():
    """Main processing function"""
    print("Setting up directories...")
    create_directories()
    
    print("\nSearching for Doppler samples...")
    sample_paths = find_doppler_samples()
    
    if not sample_paths:
        print("❌ No Doppler samples found!")
        print("\nTroubleshooting tips:")
        print("1. Check if the data path is correct:")
        print(f"   {BASE_DATA_PATH}")
        print("2. Make sure the dataset is downloaded and extracted")
        print("3. Verify the folder structure matches expected format:")
        print("   phase4_doppler/")
        print("   ├── doppler_XXXXHz/")
        print("   │   ├── snr_XXdb/")
        print("   │   │   ├── rx.npy")
        print("   │   │   └── meta.json")
        print("   │   └── ...")
        print("   └── ...")
        return
    
    print(f"\nStarting processing of {len(sample_paths)} samples...")
    results = []
    
    for sample_path in sample_paths:
        result = process_doppler_sample(sample_path)
        if result:
            results.append(result)
        print("-" * 50)
    
    # Create summary report and plots
    if results:
        print(f"\n✅ Successfully processed {len(results)} samples!")
        
        # Create summary plots
        create_summary_plots(results)
        
        # Save results summary
        summary_path = os.path.join(RESULTS_DIR, "doppler_results.json")
        
        # Convert to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'sample_id': result['sample_id'],
                'path': result['path'],
                'doppler_hz': result['doppler_hz'],
                'estimated_offset': result['estimated_offset'],
                'snr': result['snr'],
                'ber_original': result['ber_original'],
                'ber_corrected': result['ber_corrected'],
                'error_hz': result['error_hz']
            })
        
        with open(summary_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"📊 Results saved to: {summary_path}")
        
        # Print final summary
        print("\n" + "=" * 50)
        print("FINAL SUMMARY")
        print("=" * 50)
        for result in results:
            print(f"{result['sample_id']}:")
            print(f"  Doppler: {result['doppler_hz']} Hz, Est: {result['estimated_offset']:.1f} Hz")
            print(f"  Error: {result['error_hz']:.1f} Hz, SNR: {result['snr']} dB")
            if result['ber_original'] is not None:
                print(f"  BER: {result['ber_original']:.6f} → {result['ber_corrected']:.6f}")
            print()
        
    else:
        print("❌ No samples processed successfully!")
    
    print("\n" + "=" * 70)
    print("PHASE 4 PROCESSING COMPLETE!")
    print("=" * 70)

# Run the script
if __name__ == "__main__":
    main()