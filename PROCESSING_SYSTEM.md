# Spectro Data Processing System

## Overview

Sistem pemrosesan data spektrometer real-time yang terdiri dari 3 node ROS2:

1. **spectro_node** - Akuisisi data dari spectrometer
2. **spectro_processor_node** - Pemrosesan data reflectance secara real-time
3. **spectro_monitor_node** - Monitoring hasil pemrosesan

## Cara Kerja Sistem

### 1. Data Flow

```
Spectrometer → spectro_node → Save reflectance data → spectro_processor_node → Process & Plot → Publish results
```

### 2. Processing Pipeline

#### Standard Reference Processing (standard.txt):

1. Load data dari `standard.txt`
2. Filter wavelength range (400-700 nm)
3. Baseline correction (subtract 0%)
4. Smoothing (Savitzky-Golay)
5. Calculate d²R/dλ² (second derivative)
6. Divide by 100

#### Sample Data Processing:

1. Monitor folder `/home/tank/spectro_data/reflectance/`
2. Untuk setiap file baru:
   - Filter wavelength range (400-700 nm)
   - Baseline correction (subtract minimum)
   - Smoothing (Savitzky-Golay)
   - Calculate d²R/dλ² (second derivative)
   - Normalize to [0,1]
   - Calculate average value
   - Create plot dengan standard comparison
   - Save plot sebagai JPG di folder `process/`
   - Publish hasil (filename + average value)

## Installation & Setup

### Dependencies

```bash
# sudo apt install python3-pandas python3-matplotlib python3-scipy python3-watchdog
pip3 install pandas matplotlib numpy scipy watchdog
```

### Build Package

```bash
cd /home/nakanomiku/Documents/Mine/spectro_tank
colcon build --packages-select spectro
source install/setup.bash
```

## Usage

### 1. Jalankan Sistem Lengkap

```bash
ros2 launch spectro spectro_complete.launch.py
```

### 2. Jalankan Node Terpisah

#### Spectrometer Node Only

```bash
ros2 launch spectro spectro.launch.py
```

#### Processor Node Only

```bash
ros2 launch spectro spectro_processor.launch.py
```

#### Monitor Results

```bash
ros2 run spectro spectro_monitor_node
```

## ROS2 Topics

### Published by spectro_processor_node:

- `/spectro_processor_node/processing_result` (String) - JSON dengan hasil lengkap
- `/spectro_processor_node/average_derivative` (Float32) - Nilai rata-rata d²R/dλ²

### JSON Result Format:

```json
{
  "filename": "20250910_125153_353.txt",
  "average_derivative": 0.123456,
  "plot_filename": "20250910_125153_353_processed.jpg",
  "timestamp": "2025-09-10T12:51:53.353000"
}
```

## File Structure

```
/home/tank/spectro_data/
├── reflectance/
│   ├── 20250910_125153_353.txt      # Sample files
│   ├── 20250910_125200_123.txt
│   └── process/                      # Processing results
│       ├── 20250910_125153_353_processed.jpg
│       ├── 20250910_125200_123_processed.jpg
│       └── processed_files.json     # Tracking processed files
```

## Parameters

### spectro_processor_node:

- `reflectance_folder`: Path ke folder reflectance data
- `reference_file`: Path ke file reference standard
- `process_folder_name`: Nama subfolder untuk hasil processing
- `wavelength_min`: Minimum wavelength (default: 400.0)
- `wavelength_max`: Maximum wavelength (default: 700.0)
- `smoothing_window`: Window size untuk Savitzky-Golay (default: 100)
- `smoothing_order`: Polynomial order untuk smoothing (default: 2)

## Features

### ✅ Real-time Processing

- Auto-detect file baru dengan watchdog
- Process immediately saat file tersimpan

### ✅ Robust File Handling

- Support multiple format separator (tab, comma, space)
- Error handling untuk file corrupt
- Tracking processed files (tidak re-process)

### ✅ Scientific Processing

- Proper second derivative calculation
- Baseline correction
- Data normalization
- Wavelength range filtering

### ✅ Publication & Monitoring

- ROS2 topics untuk integration
- JSON format untuk structured data
- Real-time monitoring capabilities

### ✅ Visual Output

- High-quality plots (300 DPI)
- Standard vs sample comparison
- Metadata pada plot (timestamp, average value)

## Monitoring Commands

### Check Topics

```bash
ros2 topic list | grep spectro
ros2 topic echo /spectro_processor_node/processing_result
ros2 topic hz /spectro_processor_node/average_derivative
```

### Check Node Status

```bash
ros2 node list
ros2 node info /spectro_processor_node
```

## Troubleshooting

### File Not Processing

1. Check folder permissions: `ls -la /home/tank/spectro_data/reflectance/`
2. Check file format (must be tab/comma/space separated)
3. Check logs: `ros2 log --level debug`

### Plot Not Generated

1. Check process folder exists and writable
2. Check matplotlib backend
3. Verify matplotlib/scipy installation
