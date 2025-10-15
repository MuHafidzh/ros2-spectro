#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool
from std_srvs.srv import SetBool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import glob
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from datetime import datetime
import json

class SpectroProcessorNode(Node):
    def __init__(self):
        super().__init__('spectro_processor_node')
        
        # Parameters
        self.declare_parameter('reflectance_folder', '/home/tank/spectro_data/reflectance')
        self.declare_parameter('reference_file', '/home/tank/ros2_spectra_ws/standard.txt')
        self.declare_parameter('process_folder_name', 'process')
        self.declare_parameter('use_absorbance_as_primary', True)  # New flag
        
        self.reflectance_folder = self.get_parameter('reflectance_folder').value
        self.reference_file = self.get_parameter('reference_file').value
        self.process_folder_name = self.get_parameter('process_folder_name').value
        self.use_absorbance_as_primary = self.get_parameter('use_absorbance_as_primary').value
        
        # Create process folder
        self.process_folder = os.path.join(self.reflectance_folder, self.process_folder_name)
        os.makedirs(self.process_folder, exist_ok=True)
        
        # Publishers
        self.result_pub = self.create_publisher(String, '~/processing_result', 10)
        self.median_pub = self.create_publisher(Float32, '~/median_derivative', 10)  # Changed name
        
        # Service to change primary result type
        self.primary_type_service = self.create_service(
            SetBool, 
            '~/set_absorbance_primary', 
            self.set_primary_type_callback
        )
        
        # Standard data (reference processing)
        self.standard_wavelength = None
        self.standard_data = None
        self.load_and_process_standard()
        
        # Processed files tracking
        self.processed_files = set()
        self.load_processed_files_list()
        
        # File watcher
        self.setup_file_watcher()
        
        # Process existing files first
        self.process_existing_files()
        
        self.get_logger().info(f"Spectro Processor Node started")
        self.get_logger().info(f"Monitoring folder: {self.reflectance_folder}")
        self.get_logger().info(f"Process folder: {self.process_folder}")
        self.get_logger().info(f"Primary result: {'Absorbance' if self.use_absorbance_as_primary else 'Reflectance'}")

    def set_primary_result_type(self, use_absorbance=True):
        """Change the primary result type dynamically"""
        self.use_absorbance_as_primary = use_absorbance
        result_type = 'Absorbance' if use_absorbance else 'Reflectance'
        self.get_logger().info(f"Primary result changed to: {result_type}")

    def set_primary_type_callback(self, request, response):
        """Service callback to set whether to use absorbance as primary"""
        # Simply update the variable directly
        self.use_absorbance_as_primary = request.data
        
        response.success = True
        response.message = f"Primary result type set to {'absorbance' if request.data else 'reflectance'}"
        
        # Log the change
        result_type = 'absorbance' if request.data else 'reflectance'
        self.get_logger().info(f"Primary result type changed to: {result_type}")
        
        return response

    def load_processed_files_list(self):
        """Load list of already processed files"""
        processed_list_file = os.path.join(self.process_folder, 'processed_files.json')
        if os.path.exists(processed_list_file):
            try:
                with open(processed_list_file, 'r') as f:
                    self.processed_files = set(json.load(f))
                self.get_logger().info(f"Loaded {len(self.processed_files)} previously processed files")
            except Exception as e:
                self.get_logger().warn(f"Error loading processed files list: {e}")
                self.processed_files = set()
        else:
            self.processed_files = set()

    def save_processed_files_list(self):
        """Save list of processed files"""
        processed_list_file = os.path.join(self.process_folder, 'processed_files.json')
        try:
            with open(processed_list_file, 'w') as f:
                json.dump(list(self.processed_files), f)
        except Exception as e:
            self.get_logger().warn(f"Error saving processed files list: {e}")

    def read_spectrum_file(self, filepath):
        """Read a single spectrum file with GPS data from header comments"""
        gps_data = {'latitude': 0.0, 'longitude': 0.0, 'altitude': 0.0}
        try:
            # First pass: read GPS data from header comments
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# GPS_LATITUDE:'):
                        try:
                            val = line.split(':')[1].strip()
                            if val != 'nan':
                                gps_data['latitude'] = float(val)
                        except:
                            pass
                    elif line.startswith('# GPS_LONGITUDE:'):
                        try:
                            val = line.split(':')[1].strip()
                            if val != 'nan':
                                gps_data['longitude'] = float(val)
                        except:
                            pass
                    elif line.startswith('# GPS_ALTITUDE:'):
                        try:
                            val = line.split(':')[1].strip()
                            if val != 'nan':
                                gps_data['altitude'] = float(val)
                        except:
                            pass
                    elif not line.startswith('#'):
                        # Stop when we hit data
                        break
            
            # Second pass: read spectrum data, skip comments AND header row
            try:
                # Read with comment='#' to skip GPS headers
                data = pd.read_csv(filepath, sep='\t', comment='#')
                
                # Check if first row is header (contains text like "Wavelength")
                if len(data.columns) >= 2 and any('wavelength' in str(col).lower() for col in data.columns):
                    # Pandas already used first row as column names, data is ready
                    data.columns = ['wavelength', 'reflectance'] + [f'col_{i}' for i in range(2, len(data.columns))]
                else:
                    # First row might still be header, check if it contains text
                    if data.shape[0] > 0:
                        first_row = data.iloc[0]
                        if any(isinstance(val, str) and ('wavelength' in str(val).lower() or 'reflectance' in str(val).lower()) for val in first_row):
                            # Skip the header row
                            data = data.iloc[1:].reset_index(drop=True)
                    
                    # Assign column names
                    data.columns = ['wavelength', 'reflectance'] + [f'col_{i}' for i in range(2, len(data.columns))]
                    
            except:
                try:
                    # Try comma-separated
                    data = pd.read_csv(filepath, sep=',', comment='#')
                    if data.shape[0] > 0:
                        first_row = data.iloc[0]
                        if any(isinstance(val, str) and ('wavelength' in str(val).lower() or 'reflectance' in str(val).lower()) for val in first_row):
                            data = data.iloc[1:].reset_index(drop=True)
                    data.columns = ['wavelength', 'reflectance'] + [f'col_{i}' for i in range(2, len(data.columns))]
                except:
                    # Try space-separated
                    data = pd.read_csv(filepath, sep=r'\s+', comment='#')
                    if data.shape[0] > 0:
                        first_row = data.iloc[0]
                        if any(isinstance(val, str) and ('wavelength' in str(val).lower() or 'reflectance' in str(val).lower()) for val in first_row):
                            data = data.iloc[1:].reset_index(drop=True)
                    data.columns = ['wavelength', 'reflectance'] + [f'col_{i}' for i in range(2, len(data.columns))]
            
            # Check if we have enough columns
            if data.shape[1] >= 2:
                # Convert wavelength and reflectance to numeric, handling any non-numeric values
                data['wavelength'] = pd.to_numeric(data['wavelength'], errors='coerce')
                data['reflectance'] = pd.to_numeric(data['reflectance'], errors='coerce')
                
                # Drop rows with NaN values
                data = data.dropna(subset=['wavelength', 'reflectance']).reset_index(drop=True)
                
                # Keep only wavelength and reflectance columns
                data = data[['wavelength', 'reflectance']]
                
                return data['wavelength'].values, data['reflectance'].values, gps_data
            else:
                self.get_logger().error(f"File {filepath} doesn't have enough columns")
                return None, None, gps_data
                
        except Exception as e:
            self.get_logger().error(f"Error reading {filepath}: {e}")
            return None, None, gps_data

    def filter_wavelength_range(self, data, min_wl=400, max_wl=700):
        """Filter data to keep only wavelengths between min_wl and max_wl - same as process.py"""
        # Ensure wavelength is numeric
        data['wavelength'] = pd.to_numeric(data['wavelength'], errors='coerce')
        data['reflectance'] = pd.to_numeric(data['reflectance'], errors='coerce')
        
        # Drop NaN values
        data = data.dropna()
        
        # Apply filter
        mask = (data['wavelength'] >= min_wl) & (data['wavelength'] <= max_wl)
        filtered_data = data[mask].reset_index(drop=True)
        
        self.get_logger().info(f"Filtered data from {len(data)} to {len(filtered_data)} points (wavelength range: {min_wl}-{max_wl} nm)")
        
        return filtered_data

    def baseline_correction_standard(self, reflectance_data):
        """Baseline correction for standard data (subtract 0%) - same as process.py"""
        return reflectance_data - 0

    def baseline_correction_sample(self, reflectance_data):
        """Baseline correction for sample data (subtract minimum value) - same as process.py"""
        return reflectance_data - np.min(reflectance_data)

    def smooth_data(self, data, window=100, order=2):
        """Apply Savitzky-Golay smoothing filter - same as process.py"""
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
            self.get_logger().warn(f"Cannot apply smoothing, data length ({len(data)}) too small for window ({window})")
            return data
        
        return savgol_filter(data, window, order)

    def calculate_derivative(self, data, order=1, window=100, poly_order=2):
        """Calculate derivative with smoothing - same as process.py"""
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
            self.get_logger().warn(f"Cannot calculate derivative, data length ({len(data)}) too small")
            return np.gradient(data) if order == 1 else np.gradient(np.gradient(data))
        
        return savgol_filter(data, window, poly_order, deriv=order)

    def normalize_data(self, data):
        """Normalize data to [0, 1] range - same as process.py"""
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def load_and_process_standard(self):
        """Load and process standard reference data - same as process.py process_standard_data logic"""
        if not os.path.exists(self.reference_file):
            self.get_logger().error(f"Reference file not found: {self.reference_file}")
            return
        
        self.get_logger().info(f"Processing standard reference: {self.reference_file}")
        
        # Read reference file (returns GPS data too, but we ignore it for standard)
        result = self.read_spectrum_file(self.reference_file)
        if len(result) == 3:
            wavelength, reflectance, _ = result
        else:
            wavelength, reflectance = result
            
        if wavelength is None:
            self.get_logger().error("Failed to read reference file")
            return
        
        # Create dataframe for filtering
        data = pd.DataFrame({'wavelength': wavelength, 'reflectance': reflectance})
        
        # Filter wavelength range (400-700 nm)
        filtered_data = self.filter_wavelength_range(data, min_wl=400, max_wl=700)
        if len(filtered_data) == 0:
            self.get_logger().error("No data in specified wavelength range for reference")
            return
        
        # Get filtered arrays
        wavelength = filtered_data['wavelength'].values
        reflectance = filtered_data['reflectance'].values
        
        # SAME AS process.py process_standard_data:
        # Baseline correction (subtract 0%)
        corrected = self.baseline_correction_standard(reflectance)
        
        # Smoothing
        smoothed = self.smooth_data(corrected)
        
        # For standard: only smooth and divide by 100 (no derivatives)
        self.standard_data = smoothed / 100
        self.standard_wavelength = wavelength
        
        self.get_logger().info(f"Standard reference processed: {len(self.standard_data)} points")

    def process_sample_file(self, filepath):
        """Process a single sample file with dual processing paths"""
        if self.standard_data is None:
            self.get_logger().error("Standard data not available for processing")
            return
        
        filename = os.path.basename(filepath)
        self.get_logger().info(f"Processing sample file: {filename}")
        
        # Read sample data WITH GPS data
        wavelength, reflectance, gps_data = self.read_spectrum_file(filepath)
        if wavelength is None:
            self.get_logger().error(f"Failed to read sample file: {filename}")
            return
        
        # Log GPS data if available
        if gps_data['latitude'] != 0.0 or gps_data['longitude'] != 0.0:
            self.get_logger().info(f"GPS data found: Lat={gps_data['latitude']:.6f}, Lon={gps_data['longitude']:.6f}, Alt={gps_data['altitude']:.2f}m")
        
        # Create dataframe for filtering
        data = pd.DataFrame({'wavelength': wavelength, 'reflectance': reflectance})
        
        # Filter wavelength range (400-700 nm)
        filtered_data = self.filter_wavelength_range(data, min_wl=400, max_wl=700)
        if len(filtered_data) == 0:
            self.get_logger().error(f"No data in wavelength range for: {filename}")
            return
        
        # Get filtered arrays
        wavelength = filtered_data['wavelength'].values
        reflectance = filtered_data['reflectance'].values
        
        # === DUAL PROCESSING PATHS ===
        
        # PATH 1: Reflectance → Derivatives → Normalize → Median
        # Apply baseline correction and smoothing to reflectance
        corrected_refl = self.baseline_correction_sample(reflectance)
        smoothed_refl = self.smooth_data(corrected_refl)
        
        # Calculate derivatives on reflectance
        first_deriv_refl = self.calculate_derivative(smoothed_refl, order=1)
        second_deriv_refl = self.calculate_derivative(first_deriv_refl, order=1)
        third_deriv_refl = self.calculate_derivative(second_deriv_refl, order=1)
        fourth_deriv_refl = self.calculate_derivative(third_deriv_refl, order=1)
        
        # Normalize 4th derivative of reflectance
        normalized_refl = self.normalize_data(fourth_deriv_refl)
        median_refl = np.median(normalized_refl)
        
        # PATH 2: Absorbance Processing
        # Convert reflectance to absorbance: log10(1/(0.01*reflectance))
        epsilon = 1e-10
        absorbance = np.log10(1 / (0.01 * np.maximum(reflectance, epsilon)))
        
        # Apply baseline correction and smoothing to absorbance
        corrected_abs = self.baseline_correction_sample(absorbance)
        smoothed_abs = self.smooth_data(corrected_abs)
        
        # Calculate derivatives on absorbance
        first_deriv_abs = self.calculate_derivative(smoothed_abs, order=1)
        second_deriv_abs = self.calculate_derivative(first_deriv_abs, order=1)
        third_deriv_abs = self.calculate_derivative(second_deriv_abs, order=1)
        fourth_deriv_abs = self.calculate_derivative(third_deriv_abs, order=1)
        # Normalize 4th derivative of absorbance
        normalized_abs = self.normalize_data(fourth_deriv_abs)
        median_abs = np.median(normalized_abs)
        
        # Use selected path as primary result based on flag
        if self.use_absorbance_as_primary:
            primary_result = normalized_abs
            primary_median = median_abs
        else:
            primary_result = normalized_refl
            primary_median = median_refl
        
        # Interpolate standard data to match sample wavelength if needed
        if len(self.standard_wavelength) != len(wavelength) or not np.allclose(self.standard_wavelength, wavelength):
            standard_interp = np.interp(wavelength, self.standard_wavelength, self.standard_data)
        else:
            standard_interp = self.standard_data
        
        # Create plot with both processing paths
        plot_filename = self.create_dual_plot(wavelength, standard_interp, 
                                            normalized_refl, normalized_abs,
                                            filename, median_refl, median_abs, gps_data)
        
        # Publish results (using selected path as primary)
        self.publish_dual_results(filename, median_refl, median_abs, primary_median, plot_filename, gps_data)
        
        # Add to processed files
        self.processed_files.add(os.path.abspath(filepath))
        self.save_processed_files_list()

    def create_dual_plot(self, wavelength, standard_data, refl_normalized, abs_normalized, 
                        filename, median_refl, median_abs, gps_data):
        """Create and save dual processing plot"""
        plt.figure(figsize=(15, 10))
        
        # Create subplot layout
        plt.subplot(2, 1, 1)
        
        # Plot standard data
        plt.plot(wavelength, standard_data, 'k-', linewidth=2, label='Standard (smoothed/100)')
        
        # Plot reflectance processing path
        plt.plot(wavelength, refl_normalized, 'r-', linewidth=1.5, 
                label=f'Path 1: Refl→4th deriv→norm (Median: {median_refl:.6f})')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Processed Data')
        plt.title(f'Path 1 - Reflectance Processing: {filename}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(400, 700)
        
        # Second subplot for absorbance processing
        plt.subplot(2, 1, 2)
        
        # Plot standard data
        plt.plot(wavelength, standard_data, 'k-', linewidth=2, label='Standard (smoothed/100)')
        
        # Plot absorbance processing path
        plt.plot(wavelength, abs_normalized, 'b-', linewidth=1.5, 
                label=f'Path 2: Abs→4th deriv→norm (Median: {median_abs:.6f})')
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Processed Data')
        plt.title(f'Path 2 - Absorbance Processing: {filename}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(400, 700)
        
        # Add timestamp and GPS info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        info_text = f'Processed: {timestamp}\n'
        
        if gps_data['latitude'] != 0.0 or gps_data['longitude'] != 0.0:
            info_text += f'GPS: {gps_data["latitude"]:.6f}°, {gps_data["longitude"]:.6f}°\n'
            info_text += f'Alt: {gps_data["altitude"]:.2f}m'
        else:
            info_text += 'GPS: No data available'
        
        plt.figtext(0.02, 0.02, info_text, fontsize=9, alpha=0.8,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
        
        # Save plot in processing folder outside reflectance directory
        plot_filename = f"{os.path.splitext(filename)[0]}_dual_processed.jpg"
        # Create processing folder at same level as reflectance folder
        processing_folder = os.path.join(os.path.dirname(self.reflectance_folder), "processing")
        os.makedirs(processing_folder, exist_ok=True)
        plot_path = os.path.join(processing_folder, plot_filename)
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f"Dual processing plot saved: {plot_filename}")
        return plot_filename

    def publish_dual_results(self, filename, median_refl, median_abs, primary_median, plot_filename, gps_data):
        """Publish dual processing results"""
        # Publish both processing results as JSON string
        result_data = {
            'filename': filename,
            'reflectance_median': float(median_refl),
            'absorbance_median': float(median_abs),
            'primary_result': float(primary_median),  # Based on flag
            'primary_type': 'absorbance' if self.use_absorbance_as_primary else 'reflectance',
            'plot_filename': plot_filename,
            'gps_data': gps_data,
            'timestamp': datetime.now().isoformat(),
            'processing_type': 'dual_path'
        }
        
        result_msg = String()
        result_msg.data = json.dumps(result_data)
        self.result_pub.publish(result_msg)
        
        # Publish primary result based on flag
        median_msg = Float32()
        median_msg.data = float(primary_median)
        self.median_pub.publish(median_msg)
        
        primary_type = 'absorbance' if self.use_absorbance_as_primary else 'reflectance'
        self.get_logger().info(f"Published dual results for {filename}: refl_median={median_refl:.6f}, abs_median={median_abs:.6f}, primary({primary_type})={primary_median:.6f}")

    def process_existing_files(self):
        """Process any existing files that haven't been processed yet"""
        if not os.path.exists(self.reflectance_folder):
            self.get_logger().warn(f"Reflectance folder does not exist: {self.reflectance_folder}")
            return
        
        txt_files = glob.glob(os.path.join(self.reflectance_folder, "*.txt"))
        new_files = [f for f in txt_files if os.path.abspath(f) not in self.processed_files]
        
        if new_files:
            self.get_logger().info(f"Found {len(new_files)} unprocessed files")
            for filepath in sorted(new_files):
                self.process_sample_file(filepath)
        else:
            self.get_logger().info("No new files to process")

    def setup_file_watcher(self):
        """Setup file system watcher for new files"""
        class ReflectanceFileHandler(FileSystemEventHandler):
            def __init__(self, processor_node):
                self.processor_node = processor_node
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.txt'):
                    # Wait a bit to ensure file is completely written
                    time.sleep(1)
                    
                    # Check if already processed
                    if os.path.abspath(event.src_path) not in self.processor_node.processed_files:
                        self.processor_node.get_logger().info(f"New file detected: {event.src_path}")
                        self.processor_node.process_sample_file(event.src_path)
        
        self.event_handler = ReflectanceFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(self.event_handler, self.reflectance_folder, recursive=False)
        
        # Start observer in separate thread
        self.observer_thread = threading.Thread(target=self.start_file_observer, daemon=True)
        self.observer_thread.start()

    def start_file_observer(self):
        """Start the file observer"""
        try:
            self.observer.start()
            self.get_logger().info("File watcher started")
            while rclpy.ok():
                time.sleep(1)
        except Exception as e:
            self.get_logger().error(f"File watcher error: {e}")
        finally:
            self.observer.stop()
            self.observer.join()

    def destroy_node(self):
        """Clean shutdown"""
        if hasattr(self, 'observer'):
            self.observer.stop()
            self.observer.join()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = SpectroProcessorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutdown signal received")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass


if __name__ == '__main__':
    main()