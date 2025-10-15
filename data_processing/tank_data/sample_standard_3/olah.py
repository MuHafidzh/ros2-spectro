import pandas as pd
import numpy as np

# Import 5 files std_1.txt, std_2.txt, std_3.txt, std_4.txt, std_5.txt and average them
file_names = ['std_1.txt', 'std_2.txt', 'std_3.txt', 'std_4.txt', 'std_5.txt']
data_list = []

for file_name in file_names:
    print(f"Processing {file_name}...")
    try:
        # Try different separators and handle various file formats
        try:
            # First try tab-separated
            data = pd.read_csv(file_name, sep='\t', header=None)
        except:
            try:
                # Try comma-separated
                data = pd.read_csv(file_name, sep=',', header=None)
            except:
                # Try space-separated
                data = pd.read_csv(file_name, sep=r'\s+', header=None)
        
        # Skip any header rows if they contain text
        if data.shape[0] > 0:
            first_row = data.iloc[0]
            if any(isinstance(val, str) and ('wavelength' in str(val).lower() or 'intensity' in str(val).lower()) for val in first_row):
                data = data.iloc[1:].reset_index(drop=True)
        
        # Assign column names
        if data.shape[1] >= 2:
            data.columns = ['Wavelength', 'Intensity'] + [f'col_{i}' for i in range(2, data.shape[1])]
            
            # Convert to numeric, handling any non-numeric values
            data['Wavelength'] = pd.to_numeric(data['Wavelength'], errors='coerce')
            data['Intensity'] = pd.to_numeric(data['Intensity'], errors='coerce')
            
            # Drop rows with NaN values
            data = data.dropna(subset=['Wavelength', 'Intensity']).reset_index(drop=True)
            
            # Keep only wavelength and intensity columns
            data = data[['Wavelength', 'Intensity']]
            
            print(f"  Loaded {len(data)} data points")
            data_list.append(data)
        else:
            print(f"  Warning: {file_name} doesn't have enough columns")
            
    except Exception as e:
        print(f"  Error reading {file_name}: {e}")

if not data_list:
    print("No valid data files found!")
    exit(1)

print(f"\nCombining data from {len(data_list)} files...")

# Combine all data and average by wavelength
combined_data = pd.concat(data_list, ignore_index=True)

# Convert to numeric again to be sure
combined_data['Wavelength'] = pd.to_numeric(combined_data['Wavelength'], errors='coerce')
combined_data['Intensity'] = pd.to_numeric(combined_data['Intensity'], errors='coerce')

# Drop any remaining NaN values
combined_data = combined_data.dropna()

print(f"Combined data: {len(combined_data)} total points")

# Average the data by wavelength
average_data = combined_data.groupby('Wavelength', as_index=False)['Intensity'].mean()

print(f"Averaged data: {len(average_data)} unique wavelengths")
print(f"Sample wavelength range: {average_data['Wavelength'].min():.2f} - {average_data['Wavelength'].max():.2f}")

# Load coefficient file with European decimal format
print("Loading coefficient file...")

def convert_european_decimal(value):
    """Convert European decimal format (comma) to standard format (dot)"""
    if isinstance(value, str):
        return float(value.replace(',', '.'))
    return value

try:
    # Read the coefficient file as strings first
    std_koef = pd.read_csv('std_koef.txt', sep='\t', dtype=str)
    
    # Skip header if present
    if std_koef.shape[0] > 0:
        first_row = std_koef.iloc[0]
        if any('wavelength' in str(val).lower() or 'coefficient' in str(val).lower() for val in first_row):
            std_koef = std_koef.iloc[1:].reset_index(drop=True)
            print("Skipped header row")
    
    # Assign column names
    if std_koef.shape[1] >= 2:
        std_koef.columns = ['Wavelength', 'Coefficient'] + [f'col_{i}' for i in range(2, std_koef.shape[1])]
        
        # Convert European decimal format to standard format
        print("Converting European decimal format...")
        std_koef['Wavelength'] = std_koef['Wavelength'].apply(convert_european_decimal)
        std_koef['Coefficient'] = std_koef['Coefficient'].apply(convert_european_decimal)
        
        # Drop NaN values
        std_koef = std_koef.dropna(subset=['Wavelength', 'Coefficient'])
        std_koef = std_koef[['Wavelength', 'Coefficient']]
        
        print(f"Coefficient data: {len(std_koef)} points")
        print(f"Coefficient wavelength range: {std_koef['Wavelength'].min():.2f} - {std_koef['Wavelength'].max():.2f}")
    else:
        print("Error: Coefficient file doesn't have enough columns")
        exit(1)
    
except Exception as e:
    print(f"Error reading std_koef.txt: {e}")
    exit(1)

# Filter coefficient data for range 400-700 nm only
print("Filtering coefficient data for range 400-700 nm...")
std_koef_filtered = std_koef[(std_koef['Wavelength'] >= 399) & (std_koef['Wavelength'] <= 701)].copy()
print(f"Filtered coefficient data: {len(std_koef_filtered)} points")

# Use coefficient wavelengths as reference and interpolate intensity values
print("Interpolating intensity values to match coefficient wavelengths...")

# Extract wavelengths from coefficient file as target
target_wavelengths = std_koef_filtered['Wavelength'].values

# Interpolate intensity values to match coefficient wavelengths
interpolated_intensity = np.interp(target_wavelengths, 
                                  average_data['Wavelength'].values, 
                                  average_data['Intensity'].values)

# Create new dataframe with interpolated data
interpolated_data = pd.DataFrame({
    'Wavelength': target_wavelengths,
    'Intensity': interpolated_intensity
})

print(f"Interpolated data: {len(interpolated_data)} points")
print(f"Interpolated wavelength range: {interpolated_data['Wavelength'].min():.2f} - {interpolated_data['Wavelength'].max():.2f}")

# Merge with coefficient data (should be perfect match now)
print("Applying correction...")
std_olah = pd.merge(interpolated_data, std_koef_filtered, on='Wavelength', how='inner')

print(f"Merged data: {len(std_olah)} points")

if len(std_olah) == 0:
    print("Error: No data after merging!")
    exit(1)

# Apply correction
std_olah['Corrected_Intensity'] = std_olah['Intensity'] * std_olah['Coefficient']
std_olah_final = std_olah[['Wavelength', 'Corrected_Intensity']]

# Save to file
std_olah_final.to_csv('std_olah.txt', sep='\t', index=False, header=False)
print(f"Processed data saved to std_olah.txt ({len(std_olah_final)} points)")

# Show some statistics
print(f"\nSummary:")
print(f"- Wavelength range used: {std_olah_final['Wavelength'].min():.2f} - {std_olah_final['Wavelength'].max():.2f} nm")
print(f"- Original intensity range: {interpolated_data['Intensity'].min():.3f} - {interpolated_data['Intensity'].max():.3f}")
print(f"- Coefficient range: {std_koef_filtered['Coefficient'].min():.6f} - {std_koef_filtered['Coefficient'].max():.6f}")
print(f"- Corrected intensity range: {std_olah_final['Corrected_Intensity'].min():.3f} - {std_olah_final['Corrected_Intensity'].max():.3f}")

# Show first few rows as example
print(f"\nFirst 10 rows of result:")
print(std_olah_final.head(10).to_string(index=False))