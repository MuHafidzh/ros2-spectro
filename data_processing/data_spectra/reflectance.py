import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import glob

def read_spectrum_file(filepath):
    """Read a single spectrum file"""
    try:
        # Try different separators
        try:
            data = pd.read_csv(filepath, sep='\t', header=None)
        except:
            try:
                data = pd.read_csv(filepath, sep=',', header=None)
            except:
                data = pd.read_csv(filepath, sep=r'\s+', header=None)
        
        # Skip header if present
        if data.shape[0] > 0:
            first_row = data.iloc[0]
            if any(isinstance(val, str) and ('wavelength' in str(val).lower() or 'intensity' in str(val).lower()) for val in first_row):
                data = data.iloc[1:].reset_index(drop=True)
        
        # Assign column names
        if data.shape[1] >= 2:
            data.columns = ['Wavelength', 'Intensity'] + [f'col_{i}' for i in range(2, data.shape[1])]
            
            # Convert to numeric
            data['Wavelength'] = pd.to_numeric(data['Wavelength'], errors='coerce')
            data['Intensity'] = pd.to_numeric(data['Intensity'], errors='coerce')
            
            # Drop NaN values
            data = data.dropna(subset=['Wavelength', 'Intensity']).reset_index(drop=True)
            
            return data[['Wavelength', 'Intensity']]
        else:
            print(f"File {filepath} doesn't have enough columns")
            return None
            
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def read_mdg_directories():
    """Read all MDG files from mdg_1, mdg_2, mdg_3 directories"""
    mdg_data = {}
    
    for mdg_dir in ['mdg_1', 'mdg_2', 'mdg_3']:
        if not os.path.exists(mdg_dir):
            print(f"Directory {mdg_dir} not found")
            continue
            
        mdg_data[mdg_dir] = {}
        
        # Read 5 files from each directory
        for i in range(1, 6):
            filename = f"{mdg_dir}_{i}.txt"
            filepath = os.path.join(mdg_dir, filename)
            
            if os.path.exists(filepath):
                data = read_spectrum_file(filepath)
                if data is not None:
                    mdg_data[mdg_dir][i] = data
                    print(f"Read {filename}: {len(data)} points")
                else:
                    print(f"Failed to read {filename}")
            else:
                print(f"File not found: {filepath}")
    
    return mdg_data

def read_reference_files():
    """Read background.txt and reference.txt files"""
    background_data = None
    reference_data = None
    
    # Read background.txt
    if os.path.exists('background.txt'):
        background_data = read_spectrum_file('background.txt')
        if background_data is not None:
            print(f"Read background.txt: {len(background_data)} points")
        else:
            print("Failed to read background.txt")
    else:
        print("background.txt not found")
    
    # Read reference.txt
    if os.path.exists('reference.txt'):
        reference_data = read_spectrum_file('reference.txt')
        if reference_data is not None:
            print(f"Read reference.txt: {len(reference_data)} points")
        else:
            print("Failed to read reference.txt")
    else:
        print("reference.txt not found")
    
    return background_data, reference_data

def calculate_reflectance(mdg_data, background_data, reference_data):
    """Calculate reflectance using formula: reflectance % = (mdg - background) / (reference - background)"""
    reflectance_data = {}
    
    if background_data is None or reference_data is None:
        print("Background or reference data missing, cannot calculate reflectance")
        return reflectance_data
    
    # Get common wavelength range
    bg_wavelengths = background_data['Wavelength'].values
    ref_wavelengths = reference_data['Wavelength'].values
    
    for mdg_dir, files in mdg_data.items():
        reflectance_data[mdg_dir] = {}
        
        for file_num, mdg_file_data in files.items():
            print(f"Calculating reflectance for {mdg_dir}_{file_num}")
            
            mdg_wavelengths = mdg_file_data['Wavelength'].values
            mdg_intensities = mdg_file_data['Intensity'].values
            
            # Find common wavelength range
            min_wl = max(bg_wavelengths.min(), ref_wavelengths.min(), mdg_wavelengths.min())
            max_wl = min(bg_wavelengths.max(), ref_wavelengths.max(), mdg_wavelengths.max())
            
            # Create common wavelength grid (use MDG wavelengths as reference)
            mask = (mdg_wavelengths >= min_wl) & (mdg_wavelengths <= max_wl)
            common_wavelengths = mdg_wavelengths[mask]
            mdg_common = mdg_intensities[mask]
            
            # Interpolate background and reference to common wavelengths
            bg_common = np.interp(common_wavelengths, bg_wavelengths, background_data['Intensity'].values)
            ref_common = np.interp(common_wavelengths, ref_wavelengths, reference_data['Intensity'].values)
            
            # Calculate reflectance: reflectance % = (mdg - background) / (reference - background)
            denominator = ref_common - bg_common
            
            # Avoid division by zero
            valid_mask = np.abs(denominator) > 1e-10
            reflectance = np.zeros_like(mdg_common)
            reflectance[valid_mask] = ((mdg_common[valid_mask] - bg_common[valid_mask]) / 
                                     denominator[valid_mask]) * 100
            
            # Create reflectance dataframe
            reflectance_df = pd.DataFrame({
                'wavelength': common_wavelengths,
                'reflectance': reflectance
            })
            
            reflectance_data[mdg_dir][file_num] = reflectance_df
            print(f"  Calculated reflectance: {len(reflectance_df)} points, range: {reflectance.min():.2f} - {reflectance.max():.2f}%")
    
    return reflectance_data

# Import functions from process.py algorithm
def filter_wavelength_range(data, min_wl=400, max_wl=700):
    """Filter data to keep only wavelengths between min_wl and max_wl - same as process.py"""
    data['wavelength'] = pd.to_numeric(data['wavelength'], errors='coerce')
    data['reflectance'] = pd.to_numeric(data['reflectance'], errors='coerce')
    data = data.dropna()
    
    mask = (data['wavelength'] >= min_wl) & (data['wavelength'] <= max_wl)
    filtered_data = data[mask].reset_index(drop=True)
    
    print(f"Filtered data from {len(data)} to {len(filtered_data)} points (wavelength range: {min_wl}-{max_wl} nm)")
    return filtered_data

def baseline_correction_sample(reflectance_data):
    """Baseline correction for sample data (subtract minimum value) - same as process.py"""
    return reflectance_data - np.min(reflectance_data)

def smooth_data(data, window=100, order=2):
    """Apply Savitzky-Golay smoothing filter - same as process.py"""
    if window >= len(data):
        window = len(data) - 1 if len(data) > 1 else 1
    
    if window % 2 == 0:
        window -= 1
    
    if window < order + 1:
        window = order + 1
        if window % 2 == 0:
            window += 1
    
    if window >= len(data):
        print(f"Warning: Cannot apply smoothing, data length ({len(data)}) too small for window ({window})")
        return data
    
    return savgol_filter(data, window, order)

def calculate_derivative(data, order=1, window=100, poly_order=2):
    """Calculate derivative with smoothing - same as process.py"""
    if window >= len(data):
        window = len(data) - 1 if len(data) > 1 else 1
    
    if window % 2 == 0:
        window -= 1
    
    if window < poly_order + 1:
        window = poly_order + 1
        if window % 2 == 0:
            window += 1
    
    if window >= len(data):
        print(f"Warning: Cannot calculate derivative, data length ({len(data)}) too small")
        return np.gradient(data) if order == 1 else np.gradient(np.gradient(data))
    
    return savgol_filter(data, window, poly_order, deriv=order)

def normalize_data(data):
    """Normalize data to [0, 1] range - same as process.py"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def process_reflectance_data(reflectance_data):
    """Process reflectance data using same algorithm as process.py"""
    processed_data = {}
    
    for mdg_dir, files in reflectance_data.items():
        print(f"\nProcessing {mdg_dir}...")
        processed_data[mdg_dir] = {}
        
        for file_num, reflectance_df in files.items():
            print(f"  Processing {mdg_dir}_{file_num}")
            
            # Filter wavelength range (400-700 nm)
            filtered_data = filter_wavelength_range(reflectance_df, min_wl=400, max_wl=700)
            
            if len(filtered_data) == 0:
                print(f"    No data in wavelength range for {mdg_dir}_{file_num}")
                continue
            
            wavelength = filtered_data['wavelength'].values
            reflectance = filtered_data['reflectance'].values
            
            # Apply same processing as process.py for sample data:
            # 1. Baseline correction (subtract minimum)
            corrected = baseline_correction_sample(reflectance)
            
            # 2. First smoothing
            smoothed = smooth_data(corrected)
            
            # 3. First derivative with smoothing
            first_deriv = calculate_derivative(smoothed, order=1)
            
            # 4. Second derivative with smoothing
            second_deriv = calculate_derivative(first_deriv, order=1)
            
            # 5. Normalize to [0, 1]
            normalized = normalize_data(second_deriv)
            
            processed_data[mdg_dir][file_num] = {
                'wavelength': wavelength,
                'original_reflectance': reflectance,
                'corrected': corrected,
                'smoothed': smoothed,
                'first_derivative': first_deriv,
                'second_derivative': second_deriv,
                'normalized': normalized
            }
            
            print(f"    Processed successfully: {len(wavelength)} points")
    
    return processed_data

def average_and_plot_results(processed_data):
    """Average results for each MDG directory and create plots"""
    averaged_data = {}
    
    # Average data for each MDG directory
    for mdg_dir, files in processed_data.items():
        if not files:
            continue
            
        print(f"\nAveraging data for {mdg_dir}...")
        
        # Collect all normalized data
        all_wavelengths = []
        all_normalized = []
        
        for file_num, data in files.items():
            all_wavelengths.append(data['wavelength'])
            all_normalized.append(data['normalized'])
        
        if all_wavelengths:
            # Use first wavelength array as reference (they should be the same after filtering)
            wavelength = all_wavelengths[0]
            
            # Average normalized data
            normalized_array = np.array(all_normalized)
            averaged_normalized = np.mean(normalized_array, axis=0)
            
            averaged_data[mdg_dir] = {
                'wavelength': wavelength,
                'averaged_normalized': averaged_normalized
            }
            
            print(f"  Averaged {len(all_normalized)} files for {mdg_dir}")
    
    # Create plots
    plt.figure(figsize=(15, 10))

    # Plot 1: Original reflectance data
    plt.subplot(2, 2, 1)
    colors = ['blue', 'red', 'green']
    for i, (mdg_dir, files) in enumerate(processed_data.items()):
        for file_num, data in files.items():
            # Smooth individual files before plotting
            smoothed_individual = smooth_data(data['original_reflectance'], window=101, order=2)
            alpha = 0.3 if file_num > 1 else 0.7  # Make first file more visible
            plt.plot(data['wavelength'], smoothed_individual, 
                    color=colors[i % len(colors)], alpha=alpha, linewidth=1)
        
        # Plot average of original reflectance
        if files:
            all_reflectance = [data['original_reflectance'] for data in files.values()]
            avg_reflectance = np.mean(all_reflectance, axis=0)
            avg_reflectance_sv_filter = smooth_data(avg_reflectance, window=101, order=2)
            wavelength = list(files.values())[0]['wavelength']
            plt.plot(wavelength, avg_reflectance_sv_filter, color=colors[i % len(colors)], 
                    linewidth=3, label=f'{mdg_dir} (avg)')  # Make avg line thicker
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Reflectance (%)')
    plt.title('Original Reflectance Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    
    # Plot 2: Processed (normalized second derivatives)
    plt.subplot(2, 2, 2)
    for i, (mdg_dir, data) in enumerate(averaged_data.items()):
        plt.plot(data['wavelength'], data['averaged_normalized'], 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{mdg_dir} (normalized d²R/dλ²)')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized d²R/dλ²')
    plt.title('Processed Data (Averaged)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    
    # Plot 3: Individual files comparison - SEPARATE PLOTS FOR EACH MDG
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 3 separate plots side by side

    for i, (mdg_dir, files) in enumerate(processed_data.items()):
        ax = axes[i]
        
        # Plot individual files for this MDG only
        for file_num, data in files.items():
            ax.plot(data['wavelength'], data['normalized'], 
                    color=colors[i % len(colors)], alpha=0.6, linewidth=1,
                    label=f'{mdg_dir}_{file_num}')
        
        # Plot average for this MDG
        if mdg_dir in averaged_data:
            ax.plot(averaged_data[mdg_dir]['wavelength'], 
                    averaged_data[mdg_dir]['averaged_normalized'],
                    color=colors[i % len(colors)], linewidth=3, 
                    label=f'{mdg_dir} (avg)', linestyle='--')
        
        ax.set_xlabel('Wavelength (nm)')
        ax.set_ylabel('Normalized d²R/dλ²')
        ax.set_title(f'{mdg_dir.upper()} Individual Files')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(400, 700)

    plt.tight_layout()
    
    # # Plot 4: Statistical summary
    # plt.subplot(2, 2, 4)
    # for i, (mdg_dir, data) in enumerate(averaged_data.items()):
    #     mean_val = np.mean(data['averaged_normalized'])
    #     std_val = np.std(data['averaged_normalized'])
    #     median_val = np.median(data['averaged_normalized'])
        
    #     plt.bar(i, mean_val, color=colors[i % len(colors)], alpha=0.7, 
    #            label=f'{mdg_dir}\nMean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}')
    
    # plt.xlabel('MDG Directory')
    # plt.ylabel('Average Normalized d²R/dλ²')
    # plt.title('Statistical Summary')
    # plt.xticks(range(len(averaged_data)), list(averaged_data.keys()))
    # plt.legend()
    
    # plt.tight_layout()
    plt.savefig('mdg_reflectance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return averaged_data

def main():
    """Main processing function"""
    print("=== MDG Reflectance Analysis ===\n")
    
    # Step 1: Read MDG data from directories
    print("1. Reading MDG data from directories...")
    mdg_data = read_mdg_directories()
    
    if not mdg_data:
        print("No MDG data found!")
        return
    
    # Step 2: Read background and reference files
    print("\n2. Reading background and reference files...")
    background_data, reference_data = read_reference_files()
    
    if background_data is None or reference_data is None:
        print("Cannot proceed without background and reference files!")
        return
    
    # Step 3: Calculate reflectance
    print("\n3. Calculating reflectance...")
    reflectance_data = calculate_reflectance(mdg_data, background_data, reference_data)
    
    if not reflectance_data:
        print("Failed to calculate reflectance!")
        return
    
    # Step 4: Process reflectance data using process.py algorithm
    print("\n4. Processing reflectance data...")
    processed_data = process_reflectance_data(reflectance_data)
    
    if not processed_data:
        print("Failed to process reflectance data!")
        return
    
    # Step 5: Average and plot results
    print("\n5. Creating plots and analysis...")
    averaged_data = average_and_plot_results(processed_data)
    
    # Step 6: Summary
    print("\n=== SUMMARY ===")
    for mdg_dir in ['mdg_1', 'mdg_2', 'mdg_3']:
        if mdg_dir in processed_data:
            file_count = len(processed_data[mdg_dir])
            print(f"{mdg_dir}: {file_count} files processed")
            
            if mdg_dir in averaged_data:
                avg_val = np.mean(averaged_data[mdg_dir]['averaged_normalized'])
                print(f"  Average normalized d²R/dλ²: {avg_val:.6f}")
        else:
            print(f"{mdg_dir}: No data processed")
    
    print("\nProcessing completed! Check 'mdg_reflectance_analysis.png' for results.")

if __name__ == "__main__":
    main()