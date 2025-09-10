import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import ndimage
import os
import glob

def read_spectrum_files(folder_path):
    """Read all spectrum files from a folder and return wavelength and reflectance data"""
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    data_list = []
    
    for file in sorted(files):
        try:
            # Try different separators and handle various file formats
            # First try tab-separated
            try:
                data = pd.read_csv(file, sep='\t', header=None)
            except:
                # Try comma-separated
                try:
                    data = pd.read_csv(file, sep=',', header=None)
                except:
                    # Try space-separated (fix the regex warning)
                    data = pd.read_csv(file, sep=r'\s+', header=None)
            
            # Assign column names
            if data.shape[1] >= 2:
                data.columns = ['wavelength', 'reflectance'] + [f'col_{i}' for i in range(2, data.shape[1])]
                
                # Convert wavelength and reflectance to numeric, handling any non-numeric values
                data['wavelength'] = pd.to_numeric(data['wavelength'], errors='coerce')
                data['reflectance'] = pd.to_numeric(data['reflectance'], errors='coerce')
                
                # Drop rows with NaN values
                data = data.dropna(subset=['wavelength', 'reflectance']).reset_index(drop=True)
                
                # Keep only wavelength and reflectance columns
                data = data[['wavelength', 'reflectance']]
                
                data_list.append(data)
                print(f"Successfully read {file}: {len(data)} data points")
            else:
                print(f"File {file} doesn't have enough columns")
                
        except Exception as e:
            print(f"Error reading {file}: {e}")
    
    return data_list

def filter_wavelength_range(data, min_wl=400, max_wl=700):
    """Filter data to keep only wavelengths between min_wl and max_wl"""
    # Ensure wavelength is numeric
    data['wavelength'] = pd.to_numeric(data['wavelength'], errors='coerce')
    data['reflectance'] = pd.to_numeric(data['reflectance'], errors='coerce')
    
    # Drop NaN values
    data = data.dropna()
    
    # Apply filter
    mask = (data['wavelength'] >= min_wl) & (data['wavelength'] <= max_wl)
    filtered_data = data[mask].reset_index(drop=True)
    
    print(f"Filtered data from {len(data)} to {len(filtered_data)} points (wavelength range: {min_wl}-{max_wl} nm)")
    
    return filtered_data

def baseline_correction_standard(reflectance_data):
    """Baseline correction for standard data (subtract 0%)"""
    return reflectance_data - 0

def baseline_correction_sample(reflectance_data):
    """Baseline correction for sample data (subtract minimum value)"""
    return reflectance_data - np.min(reflectance_data)

def smooth_data(data, window=100, order=2):
    """Apply Savitzky-Golay smoothing filter"""
    # Ensure window is odd and not larger than data length
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
    """Calculate derivative with smoothing"""
    # Ensure window is odd and not larger than data length
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
    """Normalize data to [0, 1] range"""
    min_val = np.min(data)
    max_val = np.max(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)

def process_standard_data(folder_path):
    """Process standard data (folder 0)"""
    print("Processing standard data...")
    
    # Read all files
    data_list = read_spectrum_files(folder_path)
    if not data_list:
        print("No data files found in standard folder")
        return None, None
    
    processed_data = []
    wavelength = None
    
    for i, data in enumerate(data_list):
        print(f"Processing standard file {i+1}/{len(data_list)}")
        
        # Filter wavelength range
        filtered_data = filter_wavelength_range(data)
        
        if len(filtered_data) == 0:
            print(f"No data in wavelength range for file {i+1}")
            continue
        
        # Store wavelength from first file
        if wavelength is None:
            wavelength = filtered_data['wavelength'].values
        
        # Baseline correction (subtract 0%)
        corrected = baseline_correction_standard(filtered_data['reflectance'].values)
        
        # Smoothing
        smoothed = smooth_data(corrected)
        
        processed_data.append(smoothed)
    
    if len(processed_data) == 0:
        print("No valid data processed for standard")
        return None, None
    
    # Average all files
    averaged_data = np.mean(processed_data, axis=0)
    
    # Divide by 100
    final_data = averaged_data / 100
    
    return wavelength, final_data

def process_sample_data(folder_path):
    """Process sample data (folders 1, 2, 3, 4)"""
    print(f"Processing sample data from {folder_path}...")
    
    # Read all files
    data_list = read_spectrum_files(folder_path)
    if not data_list:
        print(f"No data files found in {folder_path}")
        return None, None
    
    processed_data = []
    wavelength = None
    
    for i, data in enumerate(data_list):
        print(f"Processing sample file {i+1}/{len(data_list)}")
        
        # Filter wavelength range
        filtered_data = filter_wavelength_range(data)
        
        if len(filtered_data) == 0:
            print(f"No data in wavelength range for file {i+1}")
            continue
        
        # Store wavelength from first file
        if wavelength is None:
            wavelength = filtered_data['wavelength'].values
        
        # Baseline correction (subtract minimum)
        corrected = baseline_correction_sample(filtered_data['reflectance'].values)
        
        # First smoothing
        smoothed = smooth_data(corrected)
        
        # First derivative with smoothing
        first_deriv = calculate_derivative(smoothed, order=1)
        
        # Second derivative with smoothing
        second_deriv = calculate_derivative(first_deriv, order=1)
        
        processed_data.append(second_deriv)
    
    if len(processed_data) == 0:
        print("No valid data processed for sample")
        return None, None
    
    # Average all files
    averaged_data = np.mean(processed_data, axis=0)
    
    # Normalize to [0, 1]
    normalized_data = normalize_data(averaged_data)
    
    return wavelength, normalized_data

def main():
    """Main processing function"""
    base_path = "data_spectra"  # Fixed folder name
    
    # Process standard data (folder 0)
    standard_path = os.path.join(base_path, "0")
    if not os.path.exists(standard_path):
        print(f"Standard data folder not found: {standard_path}")
        return
    
    std_wavelength, std_data = process_standard_data(standard_path)
    
    if std_wavelength is None:
        print("Failed to process standard data")
        return
    
    # Process sample data (folders 1, 2, 3, 4)
    sample_data = {}
    sample_wavelengths = {}
    
    for i in range(1, 5):
        sample_path = os.path.join(base_path, str(i))
        if os.path.exists(sample_path):
            wavelength, data = process_sample_data(sample_path)
            if wavelength is not None:
                sample_wavelengths[i] = wavelength
                sample_data[i] = data
        else:
            print(f"Sample folder {i} not found: {sample_path}")
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    # Plot standard data
    plt.plot(std_wavelength, std_data, 'k-', linewidth=2, label='Standard (d²R/dλ²)')
    
    # Plot sample data
    colors = ['blue', 'red', 'green', 'orange']
    for i, color in enumerate(colors, 1):
        if i in sample_data:
            plt.plot(sample_wavelengths[i], sample_data[i], 
                    color=color, linewidth=1.5, label=f'Sample {i} (normalized d²R/dλ²)')
    
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('d²R/dλ² (normalized)')
    plt.title('Spectral Analysis: Standard vs Sample Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    
    # Save plot
    plt.savefig('spectral_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Processing completed successfully!")
    print(f"Standard data points: {len(std_data)}")
    for i in sample_data:
        print(f"Sample {i} data points: {len(sample_data[i])}")

if __name__ == "__main__":
    main()