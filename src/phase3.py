import numpy as np
import json
import os
import matplotlib.pyplot as plt

print("=" * 60)
print("STARTING PHASE 3 PROCESSING")
print("=" * 60)

# PATHS
base_path = r"C:\Communications Test\Communications-Subsystem\data\cubesat_dataset\phase3_coding"
print(f"Base path: {base_path}")

# Check if base path exists
if not os.path.exists(base_path):
    print(f"ERROR: Base path does not exist: {base_path}")
    exit()

# SAMPLE PATHS
sample_paths = [
    rf"{base_path}\convolutional\snr_8db\sample_000",
    rf"{base_path}\convolutional\snr_8db\sample_001", 
    rf"{base_path}\convolutional\snr_12db\sample_000",
    rf"{base_path}\convolutional\snr_12db\sample_001",
    rf"{base_path}\reed_solomon\snr_8db\sample_000",
    rf"{base_path}\reed_solomon\snr_8db\sample_001",
    rf"{base_path}\reed_solomon\snr_12db\sample_000", 
    rf"{base_path}\reed_solomon\snr_12db\sample_001"
]

print(f"Looking for {len(sample_paths)} sample directories...")

def process_sample(sample_path):
    """Process a single sample"""
    try:
        rx_file = os.path.join(sample_path, "rx.npy")
        meta_file = os.path.join(sample_path, "meta.json")
        
        print(f"Checking: {sample_path}")
        
        # Check if files exist
        if not os.path.exists(rx_file):
            print(f"  ❌ rx.npy not found")
            return None
        if not os.path.exists(meta_file):
            print(f"  ❌ meta.json not found")
            return None
        
        print(f"  ✅ Files found, loading data...")
        
        # Load data
        signal_data = np.load(rx_file)
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        
        # Get parameters
        sps = metadata.get('sps', 8)
        snr = metadata.get('snr_db', 10)
        true_bits = np.array(metadata['clean_bits'])
        coding_type = metadata.get('coding', 'unknown')
        
        print(f"  Coding: {coding_type}, SNR: {snr}dB, Bits: {len(true_bits)}")
        
        # Simple demodulation
        symbols = signal_data[::sps]
        demod_bits = (np.real(symbols) > 0).astype(int)
        decoded_bits = demod_bits
        
        # Calculate BER
        min_len = min(len(decoded_bits), len(true_bits))
        errors = np.sum(decoded_bits[:min_len] != true_bits[:min_len])
        ber = float(errors / min_len) if min_len > 0 else 0.0
        
        print(f"  BER: {ber:.6f} ({errors} errors out of {min_len} bits)")
        
        # Save decoded bits
        output_file = os.path.join(sample_path, "decoded_bits.npy")
        np.save(output_file, decoded_bits)
        print(f"  💾 Saved decoded_bits.npy")
        
        # Create simple plot
        try:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.scatter(np.real(symbols[:500]), np.imag(symbols[:500]), alpha=0.5, s=3)
            plt.title(f'Constellation - {coding_type}')
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            compare_len = min(40, len(true_bits), len(decoded_bits))
            plt.plot(true_bits[:compare_len], 'go-', label='True', markersize=3)
            plt.plot(decoded_bits[:compare_len], 'rx-', label='Decoded', markersize=3)
            plt.title('Bit Comparison')
            plt.legend()
            plt.grid(True)
            
            plot_file = os.path.join(sample_path, "result_plot.png")
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  📊 Plot saved: result_plot.png")
            
        except Exception as plot_error:
            print(f"  ⚠️  Could not create plot: {plot_error}")
        
        return {
            'sample': os.path.basename(sample_path),
            'coding': coding_type,
            'snr': int(snr),
            'ber': float(ber),
            'errors': int(errors)
        }
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None

# Process all samples
results = []
processed_count = 0

for sample_path in sample_paths:
    print(f"\n--- Processing Sample {processed_count + 1} ---")
    
    if os.path.exists(sample_path):
        result = process_sample(sample_path)
        if result:
            results.append(result)
            processed_count += 1
    else:
        print(f"❌ Path does not exist: {sample_path}")
    
    print("-" * 40)

print("\n" + "=" * 60)
print("PROCESSING SUMMARY")
print("=" * 60)

if results:
    print(f"✅ Successfully processed {len(results)} out of {len(sample_paths)} samples!")
    
    # Print results summary
    print("\nResults:")
    for result in results:
        print(f"  {result['sample']} ({result['coding']}, {result['snr']}dB): BER = {result['ber']:.6f}")
    
    # Create summary plot
    try:
        plt.figure(figsize=(10, 6))
        
        # Separate results by coding type
        conv_results = [r for r in results if r['coding'] == 'convolutional']
        rs_results = [r for r in results if r['coding'] == 'reed_solomon']
        
        if conv_results:
            conv_snrs = sorted(set(r['snr'] for r in conv_results))
            conv_bers = [np.mean([r['ber'] for r in conv_results if r['snr'] == snr]) for snr in conv_snrs]
            plt.semilogy(conv_snrs, conv_bers, 'bo-', label='Convolutional', markersize=8, linewidth=2)
        
        if rs_results:
            rs_snrs = sorted(set(r['snr'] for r in rs_results))
            rs_bers = [np.mean([r['ber'] for r in rs_results if r['snr'] == snr]) for snr in rs_snrs]
            plt.semilogy(rs_snrs, rs_bers, 'rs-', label='Reed-Solomon', markersize=8, linewidth=2)
        
        plt.xlabel('SNR (dB)', fontweight='bold')
        plt.ylabel('Bit Error Rate (BER)', fontweight='bold')
        plt.title('Phase 3: Error Correction Performance', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save summary plot
        summary_dir = r"C:\Communications Test\Communications-Subsystem\results\phase3"
        os.makedirs(summary_dir, exist_ok=True)
        summary_plot = os.path.join(summary_dir, "performance_summary.png")
        plt.savefig(summary_plot, dpi=120, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Summary plot saved: {summary_plot}")
        
    except Exception as summary_error:
        print(f"⚠️  Could not create summary plot: {summary_error}")
    
else:
    print("❌ No samples were processed successfully!")

print("\n" + "=" * 60)
print("PHASE 3 COMPLETE")
print("=" * 60)