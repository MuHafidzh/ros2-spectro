import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# 1. Impor Data dan Batasi Rentang Panjang Gelombang
file_path = 'reflectance_20250814_174804_255.txt'
df = pd.read_csv(file_path, sep='\t')
df.columns = ['Wavelength (nm)', 'Reflectance (%)']
df_subset = df[(df['Wavelength (nm)'] >= 400) & (df['Wavelength (nm)'] <= 700)].copy()

wavelength = df_subset['Wavelength (nm)'].values
reflectance = df_subset['Reflectance (%)'].values

# 2. Koreksi Baseline
min_reflectance = reflectance.min()
reflectance_corrected = reflectance - min_reflectance
reflectance_corrected[reflectance_corrected < 0] = 0

# 3. Smoothing menggunakan filter Savitzky-Golay
window_length = 30 # Harus bilangan ganjil, jadi 30 menjadi 31
poly_order = 2
reflectance_smoothed = savgol_filter(reflectance_corrected, window_length, poly_order)

# 4. Analisis Turunan (Derivative Analysis)
derivative_1st = savgol_filter(reflectance_corrected, window_length, poly_order, deriv=1)
derivative_2nd = savgol_filter(reflectance_corrected, window_length, poly_order, deriv=2)

# 5. Plotting Hasil
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot data asli dan yang sudah dihaluskan
ax1.plot(wavelength, reflectance_corrected, 'k-', alpha=0.5, label='Data Asli')
ax1.plot(wavelength, reflectance_smoothed, 'r-', linewidth=2, label='Data Halus (Savitzky-Golay)')
ax1.set_title('Spektrum Reflektansi (Koreksi Baseline & Smoothing)')
ax1.set_xlabel('Panjang Gelombang (nm)')
ax1.set_ylabel('Reflektansi (%)')
ax1.legend()
ax1.grid(True)

# Plot turunan pertama dan kedua
ax2.plot(wavelength, derivative_1st, 'g-', label='Turunan Pertama')
ax2.plot(wavelength, derivative_2nd, 'b-', label='Turunan Kedua')
ax2.set_title('Analisis Turunan Spektrum Reflektansi')
ax2.set_xlabel('Panjang Gelombang (nm)')
ax2.set_ylabel('Nilai Turunan')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('processed_reflectance_spectrum.png')
plt.show()