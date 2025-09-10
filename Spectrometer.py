#!/usr/bin/python3
import numpy as np
# Suppress NumPy deprecation warnings for pyqtgraph compatibility
import warnings
warnings.filterwarnings("ignore", message=".*np.float.*")
warnings.filterwarnings("ignore", message=".*np.bool.*")
warnings.filterwarnings("ignore", message=".*deprecated.*numpy.*")

# Fix numpy compatibility issues
import pyqtgraph
# Only set config options if they exist
try:
    pyqtgraph.setConfigOption('useNumba', False)
except KeyError:
    pass  # Option doesn't exist in this version

try:
    pyqtgraph.setConfigOption('crashWarning', False)
except KeyError:
    pass  # Option doesn't exist in this version

# Updated Qt imports for newer pyqtgraph versions
try:
    from pyqtgraph.Qt import QtCore, QtWidgets as QtGui, uic
    from PyQt5.QtWidgets import QApplication
    qt_app = QApplication
except ImportError:
    try:
        from pyqtgraph.Qt import QtCore, QtGui, uic
        qt_app = QtGui.QApplication
    except ImportError:
        from PyQt5 import QtCore, QtWidgets as QtGui, uic
        from PyQt5.QtWidgets import QApplication
        qt_app = QApplication

import pyqtgraph as pg
import seabreeze
seabreeze.use('cseabreeze')
from seabreeze.spectrometers import list_devices, Spectrometer
from datetime import datetime
import os
import csv
import threading
from queue import Queue
import time

import paths

class SpectraViewer():

    def __init__(self, spec=None):
        if spec is None:
            dev = list_devices()[0]
            self.spec = Spectrometer(dev)
        else:
            self.spec = spec
        self.lmbd = self.spec.wavelengths()
        self.bg = np.zeros_like(self.lmbd)
        
        # Default calibration if file doesn't exist
        calibration_path = "calibration.npy"
        if os.path.exists(calibration_path):
            self.calibration = np.load(calibration_path)
        else:
            self.calibration = np.ones_like(self.lmbd)
            print("Warning: calibration.npy not found, using default calibration")

        # Reflectance mode variables
        self.reflectance_mode = False
        self.reference_spectrum = None
        self.background_spectrum = None
        self.reflectance_data = None
        self.reflectance_step = 0  # 0: setup, 1: reference, 2: background, 3: finished

        # Electric dark correction variables (default enabled)
        self.electric_dark_correction = False
        self.electric_dark_offset = None

        # Default reference/background paths
        self.default_reference_path = "default_reference.csv"
        self.default_background_path = "default_background.csv"
        self.use_default_ref = False
        self.use_default_bg = False

        # Threading for non-blocking data acquisition
        self.data_queue = Queue()
        self.acquisition_running = True
        self.acquisition_thread = None
        
        # Plot update optimization
        self.last_plot_update = 0
        self.plot_update_interval = 0.1  # 100ms minimum between plot updates

        # Check if QApplication already exists, if not create one
        try:
            if not qt_app.instance():
                self.app = qt_app([])
            else:
                self.app = qt_app.instance()
        except:
            # Fallback for older versions
            if not QtGui.QApplication.instance():
                self.app = QtGui.QApplication([])
            else:
                self.app = QtGui.QApplication.instance()
            
        self.ui = uic.loadUi("spectrum.ui")
        
        # Initialize plots
        self.setup_plots()
        
        # Default settings
        self.set_default_settings()
        
        # Setup UI connections
        self.setup_connections()
        
        self.ui.show()
        self.reset_avg()

        # Start data acquisition thread
        self.start_acquisition_thread()

        # Setup plot update timer (slower for UI responsiveness)
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.process_data_queue)
        self.timer.start(50)  # Process queue every 50ms

        # Connect close event
        self.ui.closeEvent = self.closeEvent

        # Enable debugging for reflectance calculations
        self.debug_reflectance = False  # Set to True to enable debug output

        self.app.exec_()

    def closeEvent(self, event):
        # Stop acquisition thread
        self.acquisition_running = False
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        
        # Stop timer when closing
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        # Close spectrometer connection
        if hasattr(self, 'spec'):
            self.spec.close()
        
        # Accept the close event
        event.accept()
        
        # Quit application
        self.app.quit()

    def setup_plots(self):
        # Intensity plot
        self.plot_live = pg.PlotCurveItem()
        self.pen = pg.mkPen(color='r')
        self.ui.plot_full.addItem(self.plot_live)
        
        # Reflectance plot
        self.plot_reflectance = pg.PlotCurveItem()
        self.pen_refl = pg.mkPen(color='b')
        self.ui.plot_reflectance.addItem(self.plot_reflectance)
        
        # Setup grid with better compatibility
        try:
            self.ui.plot_full.showGrid(x=True, y=True, alpha=0.5)
            self.ui.plot_reflectance.showGrid(x=True, y=True, alpha=0.5)
        except:
            # Fallback for older pyqtgraph versions
            self.ui.plot_full.showGrid(x=True, y=True)
            self.ui.plot_reflectance.showGrid(x=True, y=True)

    def set_default_settings(self):
        # Wavelength range settings with defaults (use full range)
        self.ui.xmin.setMinimum(0)
        self.ui.xmax.setMinimum(0)
        self.ui.xmin.setMaximum(self.lmbd.max()*2)
        self.ui.xmax.setMaximum(self.lmbd.max()*2)
        self.ui.xmin.setValue(self.lmbd.min())  # Use full wavelength range
        self.ui.xmax.setValue(self.lmbd.max())  # Use full wavelength range
        
        # Integration time settings
        self.ui.integration.setMinimum(int(self.spec.integration_time_micros_limits[0]/1000.0))
        self.ui.integration.setMaximum(int(self.spec.integration_time_micros_limits[1]/1000.0))
        self.ui.integration.setValue(500)  # Default 100ms
        
        # Default averaging
        self.ui.n_average.setValue(5)  # Default 5 scans
        
        # Default boxcar width
        self.ui.boxcar_width.setValue(5)  # Default 5
        
        # Set electric dark checkbox to checked by default
        if hasattr(self.ui, 'electric_dark'):
            self.ui.electric_dark.setChecked(True)
        
        # Initialize default ref/bg checkboxes (unchecked by default)
        if hasattr(self.ui, 'use_default_ref'):
            self.ui.use_default_ref.setChecked(False)
            self.use_default_ref = False
        if hasattr(self.ui, 'use_default_bg'):
            self.ui.use_default_bg.setChecked(False)
            self.use_default_bg = False
        
        # Set initial integration time
        self.set_integration_cb()
        self.update_range_cb()
        
        # Set default plot ranges
        self.set_default_plot_ranges()
        
        # Initialize electric dark correction
        self.calculate_electric_dark_offset()

    def set_default_plot_ranges(self):
        """Set default plot ranges for intensity and reflectance"""
        # Intensity plot: 0-16000 counts, full wavelength range
        self.ui.plot_full.setYRange(0, 16000)
        self.ui.plot_full.setXRange(self.lmbd.min(), self.lmbd.max())
        
        # Reflectance plot: 0-200%, full wavelength range
        self.ui.plot_reflectance.setYRange(0, 200)
        self.ui.plot_reflectance.setXRange(self.lmbd.min(), self.lmbd.max())

    def setup_connections(self):
        # Existing connections
        self.ui.integration.valueChanged.connect(self.set_integration_cb)
        self.ui.n_average.valueChanged.connect(self.reset_avg)  # Reset when average changes
        self.ui.xmin.valueChanged.connect(self.update_range_cb)
        self.ui.xmax.valueChanged.connect(self.update_range_cb)
        self.ui.autoXY.clicked.connect(self.autoXY)
        if hasattr(self.ui, 'autoXY_reflectance'):
            self.ui.autoXY_reflectance.clicked.connect(self.autoXY)
        
        # Reflectance mode connections
        self.ui.reflectance_mode.clicked.connect(self.toggle_reflectance_mode)
        self.ui.next_step.clicked.connect(self.next_reflectance_step)
        self.ui.store_spectrum.clicked.connect(self.store_current_spectrum)
        self.ui.finish_reflectance.clicked.connect(self.finish_reflectance_setup)
        
        # Save connections
        self.ui.save_reflectance_csv.clicked.connect(lambda: self.save_reflectance_data('csv'))
        self.ui.save_reflectance_txt.clicked.connect(lambda: self.save_reflectance_data('txt'))
        self.ui.save_intensity_csv.clicked.connect(lambda: self.save_intensity_data('csv'))
        self.ui.save_intensity_txt.clicked.connect(lambda: self.save_intensity_data('txt'))
        
        # Electric dark correction connection
        if hasattr(self.ui, 'electric_dark'):
            self.ui.electric_dark.clicked.connect(self.toggle_electric_dark_correction)
        
        # Default reference/background connections
        if hasattr(self.ui, 'use_default_ref'):
            self.ui.use_default_ref.clicked.connect(self.toggle_default_ref)
        if hasattr(self.ui, 'use_default_bg'):
            self.ui.use_default_bg.clicked.connect(self.toggle_default_bg)

    def toggle_reflectance_mode(self):
        self.reflectance_mode = self.ui.reflectance_mode.isChecked()
        
        if self.reflectance_mode:
            self.reflectance_step = 0
            
            # IMPORTANT: Sync internal variables with checkbox states FIRST
            self.use_default_ref = hasattr(self.ui, 'use_default_ref') and self.ui.use_default_ref.isChecked()
            self.use_default_bg = hasattr(self.ui, 'use_default_bg') and self.ui.use_default_bg.isChecked()
            
            print(f"Reflectance mode started:")
            print(f"  Use default reference: {self.use_default_ref}")
            print(f"  Use default background: {self.use_default_bg}")
            
            # Check if defaults are selected and skip setup if both are available
            if self.use_default_ref and self.use_default_bg:
                print("BOTH DEFAULTS SELECTED - Attempting to load both and skip manual calibration...")
                # Try to load defaults
                ref_loaded = self.load_default_reference()
                bg_loaded = self.load_default_background()
                
                if ref_loaded and bg_loaded:
                    # Skip setup and go directly to measurement
                    self.ui.tab_widget.setTabEnabled(1, True)
                    self.ui.tab_widget.setCurrentIndex(1)
                    self.reflectance_step = 4
                    print("SUCCESS: Using default reference and background. Reflectance mode ready!")
                    self.ui.reflectance_setup.setVisible(False)
                    return  # IMPORTANT: Return here to skip the manual setup UI
                else:
                    print("ERROR: Could not load default files, falling back to manual calibration")
                    
            elif self.use_default_ref:
                print("ONLY DEFAULT REFERENCE SELECTED - Loading ref, will need to measure background...")
                # Load default reference, still need to measure background
                ref_loaded = self.load_default_reference()
                if ref_loaded:
                    self.reflectance_step = 2  # Skip to background measurement
                    print("SUCCESS: Loaded default reference, now need to measure background")
                else:
                    print("ERROR: Could not load default reference, will measure manually")
                    
            elif self.use_default_bg:
                print("ONLY DEFAULT BACKGROUND SELECTED - Loading bg, will need to measure reference...")
                # Load default background, still need to measure reference
                bg_loaded = self.load_default_background()
                if bg_loaded:
                    self.reflectance_step = 1  # Skip to reference measurement  
                    print("SUCCESS: Loaded default background, now need to measure reference")
                else:
                    print("ERROR: Could not load default background, will measure manually")
            else:
                print("NO DEFAULTS SELECTED - Will do full manual calibration")
            
            # Show setup UI only if we didn't skip everything
            self.ui.reflectance_setup.setVisible(True)
            self.ui.tab_widget.setCurrentIndex(0)  # Show intensity tab
            self.update_reflectance_ui()
        else:
            self.ui.reflectance_setup.setVisible(False)
            self.ui.tab_widget.setTabEnabled(1, False)
            self.reflectance_step = 0

    def update_reflectance_ui(self):
        if self.reflectance_step == 0:
            self.ui.step_label.setText("Step 1: Adjust settings and click Next")
            self.ui.next_step.setVisible(True)
            self.ui.store_spectrum.setVisible(False)
            self.ui.finish_reflectance.setVisible(False)
        elif self.reflectance_step == 1:
            self.ui.step_label.setText("Step 2: Position reference sample and click Store Reference")
            self.ui.next_step.setVisible(False)
            self.ui.store_spectrum.setVisible(True)
            self.ui.store_spectrum.setText("Store Reference")
            self.ui.finish_reflectance.setVisible(False)
        elif self.reflectance_step == 2:
            self.ui.step_label.setText("Step 3: Position background/dark and click Store Background")
            self.ui.next_step.setVisible(False)
            self.ui.store_spectrum.setVisible(True)
            self.ui.store_spectrum.setText("Store Background")
            self.ui.finish_reflectance.setVisible(False)
        elif self.reflectance_step == 3:
            self.ui.step_label.setText("Step 4: Setup complete! Click Finish to start reflectance measurement")
            self.ui.next_step.setVisible(False)
            self.ui.store_spectrum.setVisible(False)
            self.ui.finish_reflectance.setVisible(True)

    def next_reflectance_step(self):
        if self.reflectance_step == 0:
            # Apply settings
            self.ui.integration.setValue(self.ui.refl_integration.value())
            self.ui.n_average.setValue(self.ui.refl_average.value())
            # Note: boxcar width would need to be implemented in seabreeze if supported
            self.reflectance_step = 1
            self.update_reflectance_ui()

    def store_current_spectrum(self):
        if self.reflectance_step == 1:
            # Store reference spectrum with proper averaging
            current_avg = self.spectra_avg.copy() / max(self.n, 1)  # Prevent division by zero
            self.reference_spectrum = current_avg.copy()
            
            # Debug output
            print(f"Stored REFERENCE spectrum:")
            print(f"  Min intensity: {self.reference_spectrum.min():.2f}")
            print(f"  Max intensity: {self.reference_spectrum.max():.2f}")
            print(f"  Mean intensity: {self.reference_spectrum.mean():.2f}")
            print(f"  Samples averaged: {max(self.n, 1)}")
            
            if hasattr(self.ui, 'save_ref_bg') and self.ui.save_ref_bg.isChecked():
                self.save_spectrum_file(self.reference_spectrum, "reference")
            
            # Option to save as default reference
            if hasattr(self.ui, 'save_as_default_ref') and self.ui.save_as_default_ref.isChecked():
                self.save_as_default_reference(self.reference_spectrum)
            
            self.reflectance_step = 2
            self.update_reflectance_ui()
            
        elif self.reflectance_step == 2:
            # Store background spectrum with proper averaging
            current_avg = self.spectra_avg.copy() / max(self.n, 1)  # Prevent division by zero
            self.background_spectrum = current_avg.copy()
            
            # Debug output
            print(f"Stored BACKGROUND spectrum:")
            print(f"  Min intensity: {self.background_spectrum.min():.2f}")
            print(f"  Max intensity: {self.background_spectrum.max():.2f}")
            print(f"  Mean intensity: {self.background_spectrum.mean():.2f}")
            print(f"  Samples averaged: {max(self.n, 1)}")
            
            # Check if background is reasonable
            if self.background_spectrum.mean() > 100:
                print("WARNING: Background intensity is high! Check if light source is OFF or detector is covered.")
            
            if hasattr(self.ui, 'save_ref_bg') and self.ui.save_ref_bg.isChecked():
                self.save_spectrum_file(self.background_spectrum, "background")
            
            # Option to save as default background
            if hasattr(self.ui, 'save_as_default_bg') and self.ui.save_as_default_bg.isChecked():
                self.save_as_default_background(self.background_spectrum)
            
            self.reflectance_step = 3
            self.update_reflectance_ui()

    def finish_reflectance_setup(self):
        # Load default reference/background if selected
        if self.use_default_ref and self.reference_spectrum is None:
            if not self.load_default_reference():
                print("Warning: Could not load default reference, please measure one manually")
                return
        
        if self.use_default_bg and self.background_spectrum is None:
            if not self.load_default_background():
                print("Warning: Could not load default background, please measure one manually")
                return
        
        # Check if we have both reference and background
        if self.reference_spectrum is None or self.background_spectrum is None:
            print("Error: Both reference and background spectra are required for reflectance mode")
            return
        
        self.ui.reflectance_setup.setVisible(False)
        self.ui.tab_widget.setTabEnabled(1, True)  # Enable reflectance tab
        self.ui.tab_widget.setCurrentIndex(1)  # Switch to reflectance tab
        self.reflectance_step = 4  # Measurement mode
        
        # Enable debugging for the first few measurements
        self.debug_reflectance = True
        self.debug_count = 0
        print("Reflectance mode activated. You can now see live reflectance measurements.")
        print("Note: Debug output will be reduced after initial measurements.")
        
        # Auto-disable debug after 5 measurements (100 calculations)
        def auto_disable_debug():
            if hasattr(self, 'debug_count') and self.debug_count > 100:
                self.debug_reflectance = False
                print("Debug mode automatically disabled after initial measurements.")
        
        # Set timer to check and disable debug
        self.debug_timer = pg.QtCore.QTimer()
        self.debug_timer.timeout.connect(auto_disable_debug)
        self.debug_timer.start(5000)  # Check every 5 seconds

    def enable_debug(self):
        """Enable debugging output for troubleshooting"""
        self.debug_reflectance = True
        print("Debug mode enabled. Reflectance calculations will show detailed output.")

    def disable_debug(self):
        """Disable debugging output"""
        self.debug_reflectance = False
        print("Debug mode disabled.")

    def calculate_electric_dark_offset(self):
        """Calculate the electric dark offset from the spectrometer"""
        try:
            # Take a measurement to get current intensities
            intensities = self.spec.intensities()
            
            # Use first/last few pixels as dark reference (common approach)
            num_dark_pixels = 10
            
            # Try left edge
            left_dark = np.mean(intensities[:num_dark_pixels])
            
            # Try right edge  
            right_dark = np.mean(intensities[-num_dark_pixels:])
            
            # Use the lower value as it's more likely to be true dark
            self.electric_dark_offset = min(left_dark, right_dark)
            
            print(f"Electric Dark Offset calculated: {self.electric_dark_offset:.2f} counts")
            print(f"  Left edge dark: {left_dark:.2f}")
            print(f"  Right edge dark: {right_dark:.2f}")
            
            return self.electric_dark_offset
            
        except Exception as e:
            print(f"Error calculating electric dark offset: {e}")
            self.electric_dark_offset = 0
            return 0

    def apply_electric_dark_correction(self, intensities):
        """Apply electric dark correction to intensities"""
        if self.electric_dark_correction and self.electric_dark_offset is not None:
            corrected = intensities - self.electric_dark_offset
            # Ensure no negative values
            corrected = np.maximum(corrected, 0)
            return corrected
        else:
            return intensities

    def toggle_electric_dark_correction(self):
        """Toggle electric dark correction on/off"""
        self.electric_dark_correction = self.ui.electric_dark.isChecked()
        
        if self.electric_dark_correction:
            if self.electric_dark_offset is None:
                self.calculate_electric_dark_offset()
            print(f"Electric dark correction ENABLED (offset: {self.electric_dark_offset:.2f} counts)")
        else:
            print("Electric dark correction DISABLED")
        
        # Reset averaging to apply correction immediately
        self.reset_avg()

    def start_acquisition_thread(self):
        """Start the data acquisition thread"""
        self.acquisition_thread = threading.Thread(target=self.acquisition_worker, daemon=True)
        self.acquisition_thread.start()
        print("Data acquisition thread started")

    def acquisition_worker(self):
        """Background thread for data acquisition"""
        while self.acquisition_running:
            try:
                # Get fresh spectrum data
                current_spectrum = self.spec.intensities()
                
                # Apply electric dark correction if enabled
                current_spectrum = self.apply_electric_dark_correction(current_spectrum)
                
                # Put data in queue for UI thread to process
                self.data_queue.put(current_spectrum)
                
                # Sleep to control acquisition rate
                time.sleep(0.05)  # 20 FPS maximum
                
            except Exception as e:
                print(f"Error in acquisition thread: {e}")
                time.sleep(0.1)  # Wait before retrying

    def process_data_queue(self):
        """Process data from acquisition thread (runs in UI thread)"""
        # Process multiple data points per timer call for efficiency
        processed_count = 0
        max_process = 5  # Process up to 5 spectra per timer call
        
        while not self.data_queue.empty() and processed_count < max_process:
            try:
                current_spectrum = self.data_queue.get_nowait()
                
                # Accumulate data
                self.spectra_avg += current_spectrum
                self.n += 1
                
                processed_count += 1
                
                # Update plot when we have enough averages
                if self.n >= self.ui.n_average.value():
                    current_time = time.time()
                    # Throttle plot updates to improve performance
                    if current_time - self.last_plot_update > self.plot_update_interval:
                        self.update_plot_fast()
                        self.last_plot_update = current_time
                    
            except:
                break  # Queue is empty

    def update_plot_fast(self):
        """Optimized plot update method"""
        try:
            # Calculate average spectrum properly
            averaged_spectrum = self.spectra_avg / max(self.n, 1)
            
            # Apply boxcar smoothing
            processed_spectrum = self.apply_boxcar(averaged_spectrum)
            
            # Update intensity plot (fast path)
            self.plot_live.setData(x=self.lmbd, y=processed_spectrum, pen=self.pen)
            
            # Update reflectance plot if in reflectance mode
            if self.reflectance_mode and self.reflectance_step == 4:
                self.reflectance_data = self.calculate_reflectance(processed_spectrum)
                if self.reflectance_data is not None:
                    self.plot_reflectance.setData(x=self.lmbd, y=self.reflectance_data, pen=self.pen_refl)
            
            # Reset averaging after plotting
            self.reset_avg()
            
        except Exception as e:
            print(f"Warning: Error in fast plot update: {e}")

    def debug_default_states(self):
        """Debug function to check current default ref/bg states"""
        print("=== DEBUG: Default Reference/Background States ===")
        
        # Check UI checkboxes
        if hasattr(self.ui, 'use_default_ref'):
            ui_ref_checked = self.ui.use_default_ref.isChecked()
            print(f"UI use_default_ref checkbox: {ui_ref_checked}")
        else:
            print("UI use_default_ref checkbox: NOT FOUND")
            
        if hasattr(self.ui, 'use_default_bg'):
            ui_bg_checked = self.ui.use_default_bg.isChecked()
            print(f"UI use_default_bg checkbox: {ui_bg_checked}")
        else:
            print("UI use_default_bg checkbox: NOT FOUND")
        
        # Check internal variables
        print(f"Internal use_default_ref: {self.use_default_ref}")
        print(f"Internal use_default_bg: {self.use_default_bg}")
        
        # Check file existence
        ref_exists = os.path.exists(self.default_reference_path)
        bg_exists = os.path.exists(self.default_background_path)
        print(f"Default reference file exists: {ref_exists} ({self.default_reference_path})")
        print(f"Default background file exists: {bg_exists} ({self.default_background_path})")
        
        # Check spectrum status
        print(f"Reference spectrum loaded: {self.reference_spectrum is not None}")
        print(f"Background spectrum loaded: {self.background_spectrum is not None}")
        
        print("=== END DEBUG ===")

    def toggle_default_ref(self):
        """Toggle using default reference spectrum"""
        self.use_default_ref = self.ui.use_default_ref.isChecked()
        print(f"Default reference checkbox: {'CHECKED' if self.use_default_ref else 'UNCHECKED'}")
        
        if self.use_default_ref:
            # Try to load immediately to verify it works
            success = self.load_default_reference()
            if not success:
                print("WARNING: Default reference file not available, but option is enabled")
        else:
            # Clear reference spectrum if disabling default
            self.reference_spectrum = None
            print("Default reference disabled - reference spectrum cleared")
        
        # IMPORTANT: Re-evaluate reflectance mode if it's currently active
        if hasattr(self, 'reflectance_mode') and self.reflectance_mode:
            print("Re-evaluating reflectance mode due to default reference change...")
            self.reevaluate_reflectance_mode()

    def toggle_default_bg(self):
        """Toggle using default background spectrum"""
        self.use_default_bg = self.ui.use_default_bg.isChecked()
        print(f"Default background checkbox: {'CHECKED' if self.use_default_bg else 'UNCHECKED'}")
        
        if self.use_default_bg:
            # Try to load immediately to verify it works
            success = self.load_default_background()
            if not success:
                print("WARNING: Default background file not available, but option is enabled")
        else:
            # Clear background spectrum if disabling default
            self.background_spectrum = None
            print("Default background disabled - background spectrum cleared")
        
        # IMPORTANT: Re-evaluate reflectance mode if it's currently active
        if hasattr(self, 'reflectance_mode') and self.reflectance_mode:
            print("Re-evaluating reflectance mode due to default background change...")
            self.reevaluate_reflectance_mode()

    def reevaluate_reflectance_mode(self):
        """Re-evaluate reflectance mode when default settings change during active mode"""
        print("=== RE-EVALUATING REFLECTANCE MODE ===")
        
        # Check current checkbox states
        current_use_ref = hasattr(self.ui, 'use_default_ref') and self.ui.use_default_ref.isChecked()
        current_use_bg = hasattr(self.ui, 'use_default_bg') and self.ui.use_default_bg.isChecked()
        
        print(f"Current default reference: {current_use_ref}")
        print(f"Current default background: {current_use_bg}")
        print(f"Reference spectrum loaded: {self.reference_spectrum is not None}")
        print(f"Background spectrum loaded: {self.background_spectrum is not None}")
        
        # Check if we can skip to measurement mode
        if current_use_ref and current_use_bg and self.reference_spectrum is not None and self.background_spectrum is not None:
            print("BOTH DEFAULTS NOW AVAILABLE - Skipping to measurement mode!")
            self.ui.reflectance_setup.setVisible(False)
            self.ui.tab_widget.setTabEnabled(1, True)
            self.ui.tab_widget.setCurrentIndex(1)
            self.reflectance_step = 4
            
            # Enable debugging for the first few measurements
            self.debug_reflectance = True
            self.debug_count = 0
            print("Reflectance mode activated. You can now see live reflectance measurements.")
            
        elif current_use_ref and self.reference_spectrum is not None:
            # Have reference, need background
            print("Reference available - skipping to background measurement")
            self.reflectance_step = 2
            self.update_reflectance_ui()
            
        elif current_use_bg and self.background_spectrum is not None:
            # Have background, need reference
            print("Background available - skipping to reference measurement")
            self.reflectance_step = 1
            self.update_reflectance_ui()
            
        else:
            # Still need manual calibration
            print("Still need manual calibration")
            if self.reflectance_step >= 4:
                # Was in measurement mode, go back to setup
                self.ui.tab_widget.setTabEnabled(1, False)
                self.ui.tab_widget.setCurrentIndex(0)
                self.ui.reflectance_setup.setVisible(True)
                self.reflectance_step = 0
                self.update_reflectance_ui()
        
        print("=== END RE-EVALUATION ===")

    def load_default_reference(self):
        """Load default reference spectrum from file"""
        try:
            print(f"Attempting to load default reference from: {self.default_reference_path}")
            
            if not os.path.exists(self.default_reference_path):
                print(f"ERROR: Default reference file not found: {self.default_reference_path}")
                return False
                
            with open(self.default_reference_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                wavelengths = []
                intensities = []
                for row in reader:
                    if len(row) >= 2:  # Ensure we have both columns
                        wavelengths.append(float(row[0]))
                        intensities.append(float(row[1]))
                
                if len(wavelengths) == 0:
                    print("ERROR: No data found in default reference file")
                    return False
                
                # Interpolate to match current wavelength array
                self.reference_spectrum = np.interp(self.lmbd, wavelengths, intensities)
                print(f"SUCCESS: Loaded default reference spectrum from {self.default_reference_path}")
                print(f"  Data points: {len(wavelengths)}")
                print(f"  Intensity range: {self.reference_spectrum.min():.1f} - {self.reference_spectrum.max():.1f}")
                return True
                
        except Exception as e:
            print(f"ERROR loading default reference: {e}")
            self.reference_spectrum = None
            return False

    def load_default_background(self):
        """Load default background spectrum from file"""
        try:
            print(f"Attempting to load default background from: {self.default_background_path}")
            
            if not os.path.exists(self.default_background_path):
                print(f"ERROR: Default background file not found: {self.default_background_path}")
                return False
                
            with open(self.default_background_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                wavelengths = []
                intensities = []
                for row in reader:
                    if len(row) >= 2:  # Ensure we have both columns
                        wavelengths.append(float(row[0]))
                        intensities.append(float(row[1]))
                
                if len(wavelengths) == 0:
                    print("ERROR: No data found in default background file")
                    return False
                
                # Interpolate to match current wavelength array
                self.background_spectrum = np.interp(self.lmbd, wavelengths, intensities)
                print(f"SUCCESS: Loaded default background spectrum from {self.default_background_path}")
                print(f"  Data points: {len(wavelengths)}")
                print(f"  Intensity range: {self.background_spectrum.min():.1f} - {self.background_spectrum.max():.1f}")
                return True
                
        except Exception as e:
            print(f"ERROR loading default background: {e}")
            self.background_spectrum = None
            return False

    def save_as_default_reference(self, spectrum):
        """Save current spectrum as default reference"""
        try:
            with open(self.default_reference_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Wavelength (nm)', 'Intensity'])
                for i in range(len(self.lmbd)):
                    writer.writerow([self.lmbd[i], spectrum[i]])
            print(f"Saved as default reference: {self.default_reference_path}")
        except Exception as e:
            print(f"Error saving default reference: {e}")

    def save_as_default_background(self, spectrum):
        """Save current spectrum as default background"""
        try:
            with open(self.default_background_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Wavelength (nm)', 'Intensity'])
                for i in range(len(self.lmbd)):
                    writer.writerow([self.lmbd[i], spectrum[i]])
            print(f"Saved as default background: {self.default_background_path}")
        except Exception as e:
            print(f"Error saving default background: {e}")

    def save_intensity_data(self, file_format):
        """Save current intensity data"""
        if not hasattr(self, 'spectra_avg') or self.n == 0:
            print("No intensity data to save")
            return
        
        name = self.ui.savepath.text() if hasattr(self.ui, 'savepath') else 'intensity'
        if name == '':
            name = 'intensity'
        
        timestamp = datetime.today().strftime("%H%M%S_%f")
        
        # Use proper averaging
        averaged_spectrum = self.spectra_avg / max(self.n, 1)
        
        # Apply current processing (boxcar, electric dark correction)
        processed_spectrum = self.apply_boxcar(averaged_spectrum)
        
        if file_format == 'csv':
            filename = f"{paths.oceanoptics()}{name}_intensity_{timestamp}.csv"
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Wavelength (nm)', 'Intensity'])
                for i in range(len(self.lmbd)):
                    writer.writerow([self.lmbd[i], processed_spectrum[i]])
        else:  # txt format
            filename = f"{paths.oceanoptics()}{name}_intensity_{timestamp}.txt"
            with open(filename, 'w') as txtfile:
                txtfile.write('Wavelength (nm)\tIntensity\n')
                for i in range(len(self.lmbd)):
                    txtfile.write(f'{self.lmbd[i]:.2f}\t{processed_spectrum[i]:.2f}\n')
        
        print(f"Saved intensity data to {filename}")

    def calculate_reflectance(self, sample_spectrum):
        if self.reference_spectrum is not None and self.background_spectrum is not None:
            # Apply boxcar smoothing if needed
            smoothed_sample = self.apply_boxcar(sample_spectrum)
            smoothed_ref = self.apply_boxcar(self.reference_spectrum)
            smoothed_bg = self.apply_boxcar(self.background_spectrum)
            
            # Calculate reflectance: (Sample - Background) / (Reference - Background) * 100
            numerator = smoothed_sample - smoothed_bg
            denominator = smoothed_ref - smoothed_bg
            
            # Optimized debug output (less frequent)
            if hasattr(self, 'debug_reflectance') and self.debug_reflectance:
                if not hasattr(self, 'debug_count'):
                    self.debug_count = 0
                self.debug_count += 1
                
                # Only print every 50th calculation to reduce spam even more
                if self.debug_count % 50 == 0:
                    print(f"Reflectance calculation (#{self.debug_count}):")
                    print(f"  Sample: {smoothed_sample.mean():.1f}, Ref: {smoothed_ref.mean():.1f}, BG: {smoothed_bg.mean():.1f}")
                    # Safe division for debug output
                    if denominator.mean() != 0:
                        expected_refl = (numerator.mean()/denominator.mean()*100)
                        print(f"  Expected reflectance: {expected_refl:.1f}%")
            
            # Fast division with minimal error checking for performance
            safe_denominator = np.where(np.abs(denominator) > 1e-3, denominator, 1e-3)
            reflectance = (numerator / safe_denominator) * 100
            
            # Fast clipping without interpolation for better performance
            reflectance = np.clip(reflectance, 0, 200)
            
            return reflectance
        return None

    def apply_boxcar(self, spectrum):
        width = self.ui.boxcar_width.value()
        if width <= 1:
            return spectrum
        
        # Simple moving average
        kernel = np.ones(width) / width
        return np.convolve(spectrum, kernel, mode='same')

    def reset_avg(self):
        self.n = 0
        self.spectra_avg = np.zeros_like(self.lmbd)

    def autoXY(self):
        """Adaptive autoscale for both intensity and reflectance tabs"""
        current_tab = self.ui.tab_widget.currentIndex()
        
        if current_tab == 0:  # Intensity tab
            # Use full wavelength range
            self.ui.xmin.setValue(self.lmbd.min())
            self.ui.xmax.setValue(self.lmbd.max())
            self.update_range_cb()
            
            # Adaptive Y range for intensity
            if hasattr(self.plot_live, 'getData') and self.plot_live.getData()[1] is not None:
                ydata = self.plot_live.getData()[1]
                if len(ydata) > 0:
                    ymin, ymax = ydata.min(), ydata.max()
                    # Add 10% padding
                    padding = (ymax - ymin) * 0.1
                    self.ui.plot_full.setYRange(max(0, ymin - padding), ymax + padding)
                else:
                    self.ui.plot_full.setYRange(0, 16000)  # Default range
            else:
                self.ui.plot_full.setYRange(0, 16000)  # Default range
                
        else:  # Reflectance tab
            # Use full wavelength range
            self.ui.xmin.setValue(self.lmbd.min())
            self.ui.xmax.setValue(self.lmbd.max())
            self.update_range_cb()
            
            # Adaptive Y range for reflectance
            if self.reflectance_data is not None:
                ymin, ymax = self.reflectance_data.min(), self.reflectance_data.max()
                # Ensure reasonable range (0-200%)
                ymin = max(0, ymin - 10)  # 10% padding below
                ymax = min(200, ymax + 10)  # 10% padding above, max 200%
                self.ui.plot_reflectance.setYRange(ymin, ymax)
            else:
                self.ui.plot_reflectance.setYRange(0, 200)  # Default range

    def acquire(self):
        """Legacy acquire method - now using threaded acquisition"""
        # This method is kept for compatibility but the real work is done in threads
        pass

    def update_range_cb(self):
        self.ui.plot_full.setXRange(self.ui.xmin.value(), self.ui.xmax.value())
        self.ui.plot_reflectance.setXRange(self.ui.xmin.value(), self.ui.xmax.value())

    def save_spectrum_file(self, spectrum, prefix):
        name = f"{prefix}_{datetime.today().strftime('%H%M%S_%f')}"
        filepath = paths.oceanoptics() + name + ".csv"
        
        # Save as CSV
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Wavelength (nm)', 'Intensity'])
            for i in range(len(self.lmbd)):
                writer.writerow([self.lmbd[i], spectrum[i]])
        
        print(f"Saved {prefix} spectrum to {filepath}")

    def save_reflectance_data(self, file_format):
        if self.reflectance_data is None:
            print("No reflectance data to save")
            return
        
        name = self.ui.savepath.text()
        if name == '':
            name = 'reflectance'
        
        timestamp = datetime.today().strftime("%H%M%S_%f")
        filename = f"{paths.oceanoptics()}{name}_reflectance_{timestamp}"
        
        if file_format == 'csv':
            filename += '.csv'
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Wavelength (nm)', 'Reflectance (%)'])
                for i in range(len(self.lmbd)):
                    writer.writerow([self.lmbd[i], self.reflectance_data[i]])
        else:  # txt format
            filename += '.txt'
            with open(filename, 'w') as txtfile:
                txtfile.write('Wavelength (nm)\tReflectance (%)\n')
                for i in range(len(self.lmbd)):
                    txtfile.write(f'{self.lmbd[i]:.2f}\t{self.reflectance_data[i]:.4f}\n')
        
        print(f"Saved reflectance data to {filename}")

    def update_plot(self):
        """Legacy update_plot method - now using update_plot_fast"""
        # Kept for compatibility, but real work is done in update_plot_fast
        self.update_plot_fast()

    def set_integration_cb(self):
        integration_time_micros = int(self.ui.integration.value() * 1000)
        min_time, max_time = self.spec.integration_time_micros_limits
        integration_time_micros = max(min_time, min(max_time, integration_time_micros))
        self.spec.integration_time_micros(integration_time_micros)
        self.reset_avg()

if __name__ == "__main__":
    try:
        sviewer = SpectraViewer()
    except Exception as e:
        print(f"Error: {e}")
        # Try to quit any existing QApplication
        try:
            app = qt_app.instance()
            if app:
                app.quit()
        except:
            try:
                app = QtGui.QApplication.instance()
                if app:
                    app.quit()
            except:
                pass