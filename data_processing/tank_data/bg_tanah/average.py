import pandas as pd
import numpy as np


# 1. Impor Data bg_1.txt, bg_2.txt, bg_3.txt, bg_4.txt, bg_5.txt dan rata-ratakan dan simpan ke bg_average.txt
file_names = ['bg_1.txt', 'bg_2.txt', 'bg_3.txt', 'bg_4.txt', 'bg_5.txt']
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

# Save the averaged data to a new file
average_data.to_csv('bg_average.txt', sep='\t', index=False, header=False)
print(f"Averaged data saved to bg_average.txt")