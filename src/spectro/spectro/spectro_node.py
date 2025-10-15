import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy, NavSatFix
from std_msgs.msg import Float32
from std_srvs.srv import SetBool

import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*np.float.*")
warnings.filterwarnings("ignore", message=".*np.bool.*")
warnings.filterwarnings("ignore", message=".*deprecated.*numpy.*")

import pyqtgraph as pg
# Fix numpy compatibility issues
try:
    pg.setConfigOption('useNumba', False)
except KeyError:
    pass
try:
    pg.setConfigOption('crashWarning', False)
except KeyError:
    pass

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

import seabreeze
seabreeze.use('cseabreeze')
from seabreeze.spectrometers import list_devices, Spectrometer
from datetime import datetime
import os
import time
import threading
from queue import Queue

class SpectroNode(Node):
    def __init__(self):
        super().__init__('spectro_node')

        # --- Initialize PyQt Application ---
        try:
            if not qt_app.instance():
                self.app = qt_app([])
            else:
                self.app = qt_app.instance()
        except:
            if not QtGui.QApplication.instance():
                self.app = QtGui.QApplication([])
            else:
                self.app = QtGui.QApplication.instance()

        # --- ROS2 Parameters ---

        # --- ROS2 Parameters ---
        self.declare_parameter('integration_time_ms', 500)
        self.declare_parameter('scans_to_average', 10)  # Default 10
        self.declare_parameter('boxcar_width', 5)
        self.declare_parameter('save_path', '~/spectro_data')
        self.declare_parameter('joystick_intensity_button', 2)
        self.declare_parameter('joystick_reflectance_button', 3)
        self.declare_parameter('joystick_ps_button', 12)
        self.declare_parameter('joystick_start_button', 9)
        self.declare_parameter('joystick_square_button', 3)  # Square button for service toggle
        self.declare_parameter('joystick_circle_button', 1)  # Circle button for threshold

        self.integration_time_ms = self.get_parameter('integration_time_ms').value
        self.scans_to_average = self.get_parameter('scans_to_average').value
        self.boxcar_width = self.get_parameter('boxcar_width').value
        self.save_path = os.path.expanduser(self.get_parameter('save_path').value)
        self.joystick_intensity_button = self.get_parameter('joystick_intensity_button').value
        self.joystick_reflectance_button = self.get_parameter('joystick_reflectance_button').value
        self.joystick_ps_button = self.get_parameter('joystick_ps_button').value
        self.joystick_start_button = self.get_parameter('joystick_start_button').value
        self.joystick_square_button = self.get_parameter('joystick_square_button').value
        self.joystick_circle_button = self.get_parameter('joystick_circle_button').value

        # --- Spectrometer Initialization ---
        try:
            devices = list_devices()
            if not devices:
                raise RuntimeError("No spectrometer found.")
            self.spec = Spectrometer(devices[0])
            self.lmbd = self.spec.wavelengths()
            self.get_logger().info(f"Spectrometer '{self.spec.model}' connected.")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize spectrometer: {e}")
            self.destroy_node()
            return

        # --- State Variables (after spectrometer initialization) ---
        self.reference_spectrum = None
        self.background_spectrum = None
        self.matriks_spectrum = None
        self.latest_intensities = np.zeros_like(self.lmbd)
        self.reflectance_data = None
        self.absorbance_data = None  # Add absorbance data
        self.reflectance_mode = False
        self.reflectance_step = 0
        
        # Joystick button state tracking for debouncing
        self.button_states = {}
        self.last_button_time = {}
        self.button_debounce_time = 0.3  # 300ms debounce

        # Threshold management
        self.min_threshold = 0.7  # Default threshold value
        self.threshold_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.current_threshold_index = 7  # Start at 0.7
        
        # Service status
        self.current_primary_result = "absorbance"  # Default based on processor node

        self.current_gps = {
            'latitude': 0.0,
            'longitude': 0.0,
            'altitude': 0.0
        }
        
        # Averaging variables
        self.spectra_avg = np.zeros_like(self.lmbd)
        self.n = 0
        
        # Threading for data acquisition
        self.data_queue = Queue()
        self.acquisition_running = True
        self.acquisition_thread = None
        
        # Plot update optimization
        self.last_plot_update = 0
        self.plot_update_interval = 0.1

        # Load UI
        ui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "..", "..", "..", "spectrum.ui")
        if not os.path.exists(ui_path):
            # Fallback to current directory
            ui_path = "spectrum.ui"
        
        if os.path.exists(ui_path):
            self.ui = uic.loadUi(ui_path)
            self.setup_plots()
            self.set_default_settings()
            self.setup_connections()
            
            # Set initial connection status
            self.update_connection_status(True)
            
            # Check for default reference and background files and auto-enable reflectance mode
            self.check_and_load_defaults()
            
            self.ui.show()
        else:
            self.get_logger().warn("UI file not found, running without GUI")
            self.ui = None

        # --- ROS2 Subscribers ---
        self.create_subscription(Joy, '/joy', self.joy_cb, 10)
        self.create_subscription(NavSatFix, '/fix', self.gps_cb, 10)

        # --- ROS2 Publishers ---
        self.threshold_pub = self.create_publisher(Float32, '/min_threshold', 10)
        
        # --- ROS2 Service Clients ---
        self.primary_service_client = self.create_client(SetBool, '/spectro_processor_node/set_absorbance_primary')

        # --- Setup ---
        self.set_integration_time()
        self.reset_avg()
        
        # Start data acquisition thread
        self.start_acquisition_thread()
        
        # Setup plot update timer if UI is available
        if self.ui:
            self.timer = pg.QtCore.QTimer()
            self.timer.timeout.connect(self.process_data_queue)
            self.timer.start(50)
            self.ui.closeEvent = self.closeEvent
        
        # Setup threshold publishing timer
        self.threshold_timer = pg.QtCore.QTimer()
        self.threshold_timer.timeout.connect(self.publish_threshold_continuously)
        self.threshold_timer.start(1000)  # Publish every 1 second
        
        # Initial UI status update
        self.update_ui_status()
        
        # Publish initial threshold value
        self.publish_threshold_continuously()

        self.get_logger().info("Spectrometer node started.")
        self.get_logger().info(f"Data will be saved to: {self.save_path}")
        os.makedirs(self.save_path, exist_ok=True)

    def closeEvent(self, event):
        """Handle window close event"""
        self.acquisition_running = False
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        
        if hasattr(self, 'timer'):
            self.timer.stop()
        
        if hasattr(self, 'threshold_timer'):
            self.threshold_timer.stop()
        
        if hasattr(self, 'spec'):
            self.spec.close()
        
        event.accept()
        self.destroy_node()
        rclpy.shutdown()

    def gps_cb(self, msg):
        """Callback for GPS NavSatFix messages"""
        self.current_gps['latitude'] = msg.latitude
        self.current_gps['longitude'] = msg.longitude
        self.current_gps['altitude'] = msg.altitude


    def setup_plots(self):
        """Setup pyqtgraph plots"""
        if not self.ui:
            return
            
        # Intensity plot
        self.plot_live = pg.PlotCurveItem()
        self.pen = pg.mkPen(color='r')
        self.ui.plot_full.addItem(self.plot_live)
        
        # Reflectance plot
        self.plot_reflectance = pg.PlotCurveItem()
        self.pen_refl = pg.mkPen(color='b')
        self.ui.plot_reflectance.addItem(self.plot_reflectance)
        
        # Absorbance plot
        self.plot_absorbance = pg.PlotCurveItem()
        self.pen_abs = pg.mkPen(color='g')
        self.ui.plot_absorbance.addItem(self.plot_absorbance)
        
        # Setup grid
        try:
            self.ui.plot_full.showGrid(x=True, y=True, alpha=0.5)
            self.ui.plot_reflectance.showGrid(x=True, y=True, alpha=0.5)
            self.ui.plot_absorbance.showGrid(x=True, y=True, alpha=0.5)
        except:
            self.ui.plot_full.showGrid(x=True, y=True)
            self.ui.plot_reflectance.showGrid(x=True, y=True)
            self.ui.plot_absorbance.showGrid(x=True, y=True)

    def set_default_settings(self):
        """Set default UI settings"""
        if not self.ui:
            return
            
        # Wavelength range settings
        self.ui.xmin.setMinimum(0)
        self.ui.xmax.setMinimum(0)
        self.ui.xmin.setMaximum(self.lmbd.max()*2)
        self.ui.xmax.setMaximum(self.lmbd.max()*2)
        self.ui.xmin.setValue(self.lmbd.min())
        self.ui.xmax.setValue(self.lmbd.max())
        
        # Integration time settings
        self.ui.integration.setMinimum(int(self.spec.integration_time_micros_limits[0]/1000.0))
        self.ui.integration.setMaximum(int(self.spec.integration_time_micros_limits[1]/1000.0))
        self.ui.integration.setValue(self.integration_time_ms)
        
        # Default averaging
        self.ui.n_average.setValue(self.scans_to_average)
        
        # Default boxcar width
        self.ui.boxcar_width.setValue(self.boxcar_width)
        
        # Set default plot ranges
        self.set_default_plot_ranges()

    def set_default_plot_ranges(self):
        """Set default plot ranges"""
        if not self.ui:
            return
            
        self.ui.plot_full.setYRange(0, 16000)
        self.ui.plot_full.setXRange(self.lmbd.min(), self.lmbd.max())
        self.ui.plot_reflectance.setYRange(0, 200)
        self.ui.plot_reflectance.setXRange(self.lmbd.min(), self.lmbd.max())
        self.ui.plot_absorbance.setYRange(0, 2.0)  # Typical absorbance range
        self.ui.plot_absorbance.setXRange(self.lmbd.min(), self.lmbd.max())

    def setup_connections(self):
        """Setup UI signal connections"""
        if not self.ui:
            return
            
        # Basic connections
        self.ui.integration.valueChanged.connect(self.set_integration_cb)
        self.ui.n_average.valueChanged.connect(self.reset_avg)
        self.ui.xmin.valueChanged.connect(self.update_range_cb)
        self.ui.xmax.valueChanged.connect(self.update_range_cb)
        self.ui.autoXY.clicked.connect(self.autoXY)
        if hasattr(self.ui, 'autoXY_reflectance'):
            self.ui.autoXY_reflectance.clicked.connect(self.autoXY)
        if hasattr(self.ui, 'autoXY_absorbance'):
            self.ui.autoXY_absorbance.clicked.connect(self.autoXY)
        
        # Reflectance mode connections
        self.ui.reflectance_mode.clicked.connect(self.toggle_reflectance_mode)
        self.ui.next_step.clicked.connect(self.next_reflectance_step)
        self.ui.store_spectrum.clicked.connect(self.store_current_spectrum)
        self.ui.finish_reflectance.clicked.connect(self.finish_reflectance_setup)
        
        # Save connections (only TXT format)
        self.ui.save_reflectance_txt.clicked.connect(lambda: self.save_reflectance_data('txt'))
        self.ui.save_intensity_txt.clicked.connect(lambda: self.save_intensity_data('txt'))
        if hasattr(self.ui, 'save_absorbance_txt'):
            self.ui.save_absorbance_txt.clicked.connect(lambda: self.save_absorbance_data('txt'))
        
        # Hide CSV buttons since we only use TXT
        if hasattr(self.ui, 'save_reflectance_csv'):
            self.ui.save_reflectance_csv.setVisible(False)
        if hasattr(self.ui, 'save_intensity_csv'):
            self.ui.save_intensity_csv.setVisible(False)
        if hasattr(self.ui, 'save_absorbance_csv'):
            self.ui.save_absorbance_csv.setVisible(False)

    def start_acquisition_thread(self):
        """Start background data acquisition thread"""
        self.acquisition_thread = threading.Thread(target=self.acquisition_worker, daemon=True)
        self.acquisition_thread.start()
        self.get_logger().info("Data acquisition thread started")

    def acquisition_worker(self):
        """Background thread for continuous data acquisition"""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self.acquisition_running:
            try:
                if not hasattr(self, 'spec') or not self.spec:
                    time.sleep(0.5)
                    continue
                    
                current_spectrum = self.spec.intensities()
                self.data_queue.put(current_spectrum)
                consecutive_errors = 0  # Reset error counter on success
                time.sleep(0.05)  # 20 FPS max
            except Exception as e:
                consecutive_errors += 1
                self.get_logger().warn(f"Error in acquisition thread: {e}")
                
                # If too many consecutive errors, assume disconnection
                if consecutive_errors >= max_consecutive_errors:
                    self.get_logger().error("Too many acquisition errors, assuming spectrometer disconnected")
                    if self.ui:
                        self.update_connection_status(False)
                    consecutive_errors = 0  # Reset to avoid spam
                
                time.sleep(0.1)

    def process_data_queue(self):
        """Process data from acquisition thread in UI thread"""
        if not self.ui:
            # If no UI, still need to process spectrum data
            try:
                if not self.data_queue.empty():
                    current_spectrum = self.data_queue.get_nowait()
                    self.latest_intensities = self.apply_boxcar(current_spectrum)
            except:
                pass
            return
            
        processed_count = 0
        max_process = 5
        
        while not self.data_queue.empty() and processed_count < max_process:
            try:
                current_spectrum = self.data_queue.get_nowait()
                self.spectra_avg += current_spectrum
                self.n += 1
                processed_count += 1
                
                if self.n >= self.ui.n_average.value():
                    current_time = time.time()
                    if current_time - self.last_plot_update > self.plot_update_interval:
                        self.update_plot_fast()
                        self.last_plot_update = current_time
            except:
                break

    def update_plot_fast(self):
        """Fast plot update method"""
        if not self.ui:
            return
            
        try:
            averaged_spectrum = self.spectra_avg / max(self.n, 1)
            processed_spectrum = self.apply_boxcar(averaged_spectrum)
            self.latest_intensities = processed_spectrum
            
            # Update intensity plot
            self.plot_live.setData(x=self.lmbd, y=processed_spectrum, pen=self.pen)
            
            # Update reflectance and absorbance plots if in reflectance mode
            if self.reflectance_mode and self.reflectance_step == 4:
                self.reflectance_data = self.calculate_reflectance(processed_spectrum)
                if self.reflectance_data is not None:
                    self.plot_reflectance.setData(x=self.lmbd, y=self.reflectance_data, pen=self.pen_refl)
                    
                    # Auto-rescale reflectance plot
                    ymin, ymax = self.reflectance_data.min(), self.reflectance_data.max()
                    padding = (ymax - ymin) * 0.1 if ymax != ymin else 1.0
                    self.ui.plot_reflectance.setYRange(ymin - padding, ymax + padding)
                
                # Update absorbance plot if absorbance data is available
                if self.absorbance_data is not None:
                    self.plot_absorbance.setData(x=self.lmbd, y=self.absorbance_data, pen=self.pen_abs)
                    
                    # Auto-rescale absorbance plot
                    ymin, ymax = self.absorbance_data.min(), self.absorbance_data.max()
                    padding = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
                    self.ui.plot_absorbance.setYRange(ymin - padding, ymax + padding)
            
            # Reset averaging
            self.reset_avg()
        except Exception as e:
            self.get_logger().warn(f"Error in plot update: {e}")

    def toggle_reflectance_mode(self):
        """Toggle reflectance mode on/off"""
        if not self.ui:
            return
            
        self.reflectance_mode = self.ui.reflectance_mode.isChecked()
        
        if self.reflectance_mode:
            self.reflectance_step = 0
            self.ui.reflectance_setup.setVisible(True)
            self.ui.tab_widget.setCurrentIndex(0)
            self.update_reflectance_ui()
        else:
            self.ui.reflectance_setup.setVisible(False)
            self.ui.tab_widget.setTabEnabled(1, False)  # Disable reflectance tab
            self.ui.tab_widget.setTabEnabled(2, False)  # Disable absorbance tab
            self.reflectance_step = 0

    def update_reflectance_ui(self):
        """Update reflectance setup UI based on current step"""
        if not self.ui:
            return
            
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
        """Move to next step in reflectance setup"""
        if not self.ui or self.reflectance_step != 0:
            return
            
        # Apply settings
        self.ui.integration.setValue(self.ui.refl_integration.value())
        self.ui.n_average.setValue(self.ui.refl_average.value())
        self.reflectance_step = 1
        self.update_reflectance_ui()

    def store_current_spectrum(self):
        """Store reference or background spectrum"""
        if not self.ui:
            return
            
        if self.reflectance_step == 1:
            # Store reference
            current_avg = self.spectra_avg.copy() / max(self.n, 1)
            self.reference_spectrum = current_avg.copy()
            self.get_logger().info(f"Reference spectrum stored (mean: {self.reference_spectrum.mean():.1f})")
            
            if hasattr(self.ui, 'save_ref_bg') and self.ui.save_ref_bg.isChecked():
                self.save_spectrum_to_file(self.reference_spectrum, "ref")
            
            self.reflectance_step = 2
            self.update_reflectance_ui()
            
        elif self.reflectance_step == 2:
            # Store background
            current_avg = self.spectra_avg.copy() / max(self.n, 1)
            self.background_spectrum = current_avg.copy()
            self.get_logger().info(f"Background spectrum stored (mean: {self.background_spectrum.mean():.1f})")
            
            if hasattr(self.ui, 'save_ref_bg') and self.ui.save_ref_bg.isChecked():
                self.save_spectrum_to_file(self.background_spectrum, "bg")
            
            self.reflectance_step = 3
            self.update_reflectance_ui()

    def finish_reflectance_setup(self):
        """Finish reflectance setup and start measurement mode"""
        if not self.ui:
            return
            
        if self.reference_spectrum is None or self.background_spectrum is None:
            self.get_logger().error("Both reference and background spectra required")
            return
        
        self.ui.reflectance_setup.setVisible(False)
        self.ui.tab_widget.setTabEnabled(1, True)  # Enable reflectance tab
        self.ui.tab_widget.setTabEnabled(2, True)  # Enable absorbance tab
        self.ui.tab_widget.setCurrentIndex(1)
        self.reflectance_step = 4
        self.get_logger().info("Reflectance measurement mode activated")

    def autoXY(self):
        """Auto-scale plots"""
        if not self.ui:
            return
            
        current_tab = self.ui.tab_widget.currentIndex()
        
        if current_tab == 0:  # Intensity tab
            self.ui.xmin.setValue(self.lmbd.min())
            self.ui.xmax.setValue(self.lmbd.max())
            self.update_range_cb()
            
            if hasattr(self.plot_live, 'getData') and self.plot_live.getData()[1] is not None:
                ydata = self.plot_live.getData()[1]
                if len(ydata) > 0:
                    ymin, ymax = ydata.min(), ydata.max()
                    padding = (ymax - ymin) * 0.1
                    self.ui.plot_full.setYRange(max(0, ymin - padding), ymax + padding)
                else:
                    self.ui.plot_full.setYRange(0, 16000)
            else:
                self.ui.plot_full.setYRange(0, 16000)
        elif current_tab == 1:  # Reflectance tab
            self.ui.xmin.setValue(self.lmbd.min())
            self.ui.xmax.setValue(self.lmbd.max())
            self.update_range_cb()
            
            if self.reflectance_data is not None:
                ymin, ymax = self.reflectance_data.min(), self.reflectance_data.max()
                ymin = max(0, ymin - 10)
                ymax = min(200, ymax + 10)
                self.ui.plot_reflectance.setYRange(ymin, ymax)
            else:
                self.ui.plot_reflectance.setYRange(0, 200)
        elif current_tab == 2:  # Absorbance tab
            self.ui.xmin.setValue(self.lmbd.min())
            self.ui.xmax.setValue(self.lmbd.max())
            self.update_range_cb()
            
            if self.absorbance_data is not None:
                ymin, ymax = self.absorbance_data.min(), self.absorbance_data.max()
                padding = (ymax - ymin) * 0.1
                self.ui.plot_absorbance.setYRange(ymin - padding, ymax + padding)
            else:
                self.ui.plot_absorbance.setYRange(0, 2.0)

    def update_range_cb(self):
        """Update plot ranges"""
        if not self.ui:
            return
            
        self.ui.plot_full.setXRange(self.ui.xmin.value(), self.ui.xmax.value())
        self.ui.plot_reflectance.setXRange(self.ui.xmin.value(), self.ui.xmax.value())
        self.ui.plot_absorbance.setXRange(self.ui.xmin.value(), self.ui.xmax.value())

    def update_connection_status(self, connected):
        """Update connection status display on UI"""
        if not self.ui or not hasattr(self.ui, 'connection_status'):
            return
        
        if connected:
            self.ui.connection_status.setText("Status: Connected")
            self.ui.connection_status.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        else:
            self.ui.connection_status.setText("Status: Disconnected")
            self.ui.connection_status.setStyleSheet("QLabel { color: red; font-weight: bold; }")

    def clear_plots(self):
        """Clear all plot data"""
        if not self.ui:
            return
        
        try:
            # Clear intensity plot
            self.plot_live.setData(x=[], y=[])
            
            # Clear reflectance plot
            self.plot_reflectance.setData(x=[], y=[])
            
            # Clear absorbance plot
            self.plot_absorbance.setData(x=[], y=[])
            
            # Reset data arrays
            if hasattr(self, 'lmbd'):
                self.latest_intensities = np.zeros_like(self.lmbd)
                self.spectra_avg = np.zeros_like(self.lmbd)
            else:
                self.latest_intensities = np.array([])
                self.spectra_avg = np.array([])
            
            self.reflectance_data = None
            self.absorbance_data = None
            self.n = 0
            
            self.get_logger().info("Plots cleared due to disconnection")
        except Exception as e:
            self.get_logger().warn(f"Error clearing plots: {e}")

    def toggle_tab_mode(self):
        """Toggle between intensity, reflectance, and absorbance tabs"""
        if not self.ui:
            return
        
        current_index = self.ui.tab_widget.currentIndex()
        
        if current_index == 0:  # Currently on intensity tab
            # Switch to reflectance tab if enabled
            if self.ui.tab_widget.isTabEnabled(1):
                self.ui.tab_widget.setCurrentIndex(1)
                self.get_logger().info("Switched to Reflectance tab")
            else:
                self.get_logger().warn("Reflectance tab not available - enable reflectance mode first")
        elif current_index == 1:  # Currently on reflectance tab
            # Switch to absorbance tab if enabled
            if self.ui.tab_widget.isTabEnabled(2):
                self.ui.tab_widget.setCurrentIndex(2)
                self.get_logger().info("Switched to Absorbance tab")
            else:
                # Fall back to intensity tab
                self.ui.tab_widget.setCurrentIndex(0)
                self.get_logger().info("Switched to Intensity tab")
        else:  # Currently on absorbance tab (index 2)
            # Switch back to intensity tab
            self.ui.tab_widget.setCurrentIndex(0)
            self.get_logger().info("Switched to Intensity tab")

    def toggle_spectrometer_connection(self):
        """Connect or disconnect spectrometer"""
        try:
            if hasattr(self, 'spec') and self.spec:
                # Disconnect
                self.acquisition_running = False
                if self.acquisition_thread and self.acquisition_thread.is_alive():
                    self.acquisition_thread.join(timeout=1.0)
                self.spec.close()
                self.spec = None
                
                # Clear plots and update UI
                self.clear_plots()
                self.update_connection_status(False)
                
                self.get_logger().info("Spectrometer disconnected")
            else:
                # Connect
                devices = list_devices()
                if not devices:
                    self.get_logger().error("No spectrometer found for connection")
                    return
                
                self.spec = Spectrometer(devices[0])
                self.lmbd = self.spec.wavelengths()
                self.latest_intensities = np.zeros_like(self.lmbd)
                self.spectra_avg = np.zeros_like(self.lmbd)
                
                # Restart acquisition
                self.acquisition_running = True
                self.start_acquisition_thread()
                
                # Reset integration time
                self.set_integration_time()
                
                # Update UI
                self.update_connection_status(True)
                
                self.get_logger().info(f"Spectrometer '{self.spec.model}' reconnected")
                
        except Exception as e:
            self.get_logger().error(f"Error toggling spectrometer connection: {e}")
            # Make sure UI reflects disconnected state on error
            self.update_connection_status(False)

    def cycle_threshold_value(self):
        """Cycle through threshold values: 0.7->0.8->0.9->0.0->0.1->...->0.6->0.7"""
        self.current_threshold_index = (self.current_threshold_index + 1) % len(self.threshold_values)
        self.min_threshold = self.threshold_values[self.current_threshold_index]
        
        # Publish the new threshold value
        threshold_msg = Float32()
        threshold_msg.data = self.min_threshold
        self.threshold_pub.publish(threshold_msg)
        
        self.get_logger().info(f"Threshold changed to: {self.min_threshold}")
        
        # Update UI if available
        self.update_ui_status()

    def toggle_primary_result_service(self):
        """Toggle between absorbance and reflectance as primary result via service"""
        if not self.primary_service_client.service_is_ready():
            self.get_logger().warn("Primary result service not available")
            return
        
        # Toggle: if current is absorbance, switch to reflectance, and vice versa
        new_use_absorbance = False if self.current_primary_result == "absorbance" else True
        
        request = SetBool.Request()
        request.data = new_use_absorbance
        
        future = self.primary_service_client.call_async(request)
        future.add_done_callback(self.service_response_callback)

    def service_response_callback(self, future):
        """Handle service response"""
        try:
            response = future.result()
            if response.success:
                # Update local status based on service response
                if "absorbance" in response.message:
                    self.current_primary_result = "absorbance"
                else:
                    self.current_primary_result = "reflectance"
                
                self.get_logger().info(f"Service call successful: {response.message}")
                self.update_ui_status()
            else:
                self.get_logger().error(f"Service call failed: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Service call exception: {e}")

    def update_ui_status(self):
        """Update UI status display"""
        if not self.ui or not hasattr(self.ui, 'connection_status'):
            return
        
        # Update status text with primary result and threshold
        status_text = f"Primary: {self.current_primary_result.title()} | Threshold: {self.min_threshold}"
        self.ui.connection_status.setText(status_text)

    def publish_threshold_continuously(self):
        """Publish current threshold value (called periodically)"""
        threshold_msg = Float32()
        threshold_msg.data = self.min_threshold
        self.threshold_pub.publish(threshold_msg)

    def is_button_pressed(self, button_index, current_state):
        """Check if button is pressed with debouncing"""
        current_time = time.time()
        
        # Initialize button state if not exists
        if button_index not in self.button_states:
            self.button_states[button_index] = False
            self.last_button_time[button_index] = 0
        
        # Check for button press (transition from 0 to 1)
        if current_state == 1 and not self.button_states[button_index]:
            # Check debounce time
            if current_time - self.last_button_time[button_index] > self.button_debounce_time:
                self.button_states[button_index] = True
                self.last_button_time[button_index] = current_time
                return True
        elif current_state == 0:
            self.button_states[button_index] = False
        
        return False

    def reset_avg(self):
        """Reset averaging"""
        self.n = 0
        self.spectra_avg = np.zeros_like(self.lmbd)

    def set_integration_cb(self):
        """Set integration time from UI"""
        if not self.ui:
            return
            
        integration_time_micros = int(self.ui.integration.value() * 1000)
        min_time, max_time = self.spec.integration_time_micros_limits
        integration_time_micros = max(min_time, min(max_time, integration_time_micros))
        self.spec.integration_time_micros(integration_time_micros)
        self.integration_time_ms = integration_time_micros / 1000
        self.reset_avg()

    def save_intensity_data(self, file_format):
        """Save intensity data to file"""
        # Check if spectrometer is connected
        if not hasattr(self, 'spec') or not self.spec:
            self.get_logger().warn("Cannot save intensity data: Spectrometer not connected")
            return
            
        # Use latest intensities if no averaged data available
        if hasattr(self, 'spectra_avg') and self.n > 0:
            averaged_spectrum = self.spectra_avg / max(self.n, 1)
            processed_spectrum = self.apply_boxcar(averaged_spectrum)
        elif hasattr(self, 'latest_intensities') and len(self.latest_intensities) > 0:
            processed_spectrum = self.latest_intensities
        else:
            self.get_logger().warn("No intensity data to save")
            return
        
        # Create intensity subfolder
        intensity_folder = os.path.join(self.save_path, "intensity")
        os.makedirs(intensity_folder, exist_ok=True)
        
        name = self.ui.savepath.text() if self.ui and hasattr(self.ui, 'savepath') and self.ui.savepath.text() else ''
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if name:
            filename = os.path.join(intensity_folder, f"{name}_{timestamp}.txt")
        else:
            filename = os.path.join(intensity_folder, f"{timestamp}.txt")
        
        try:
            with open(filename, 'w') as txtfile:
                txtfile.write('Wavelength (nm)\tIntensity\n')
                for i in range(len(self.lmbd)):
                    txtfile.write(f'{self.lmbd[i]:.2f}\t{processed_spectrum[i]:.2f}\n')
            self.get_logger().info(f"Saved intensity data to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save intensity data: {e}")

    def save_reflectance_data(self, file_format):
        """Save reflectance data to file"""
        # Check if spectrometer is connected
        if not hasattr(self, 'spec') or not self.spec:
            self.get_logger().warn("Cannot save reflectance data: Spectrometer not connected")
            return
            
        if self.reflectance_data is None:
            self.get_logger().warn("No reflectance data to save")
            return
        
        # Create reflectance subfolder
        reflectance_folder = os.path.join(self.save_path, "reflectance")
        os.makedirs(reflectance_folder, exist_ok=True)
        
        name = self.ui.savepath.text() if self.ui and hasattr(self.ui, 'savepath') and self.ui.savepath.text() else ''
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if name:
            filename = os.path.join(reflectance_folder, f"{name}_{timestamp}.txt")
        else:
            filename = os.path.join(reflectance_folder, f"{timestamp}.txt")
        
        try:
            with open(filename, 'w') as txtfile:
                txtfile.write(f'# GPS_LATITUDE: {self.current_gps["latitude"]:.6f}\n')
                txtfile.write(f'# GPS_LONGITUDE: {self.current_gps["longitude"]:.6f}\n')
                txtfile.write(f'# GPS_ALTITUDE: {self.current_gps["altitude"]:.2f}\n')
                txtfile.write('Wavelength (nm)\tReflectance (%)\n')
                for i in range(len(self.lmbd)):
                    txtfile.write(f'{self.lmbd[i]:.2f}\t{self.reflectance_data[i]:.4f}\n')
            self.get_logger().info(f"Saved reflectance data to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save reflectance data: {e}")

    def save_absorbance_data(self, file_format):
        """Save absorbance data to file"""
        # Check if spectrometer is connected
        if not hasattr(self, 'spec') or not self.spec:
            self.get_logger().warn("Cannot save absorbance data: Spectrometer not connected")
            return
            
        if self.absorbance_data is None:
            self.get_logger().warn("No absorbance data to save")
            return
        
        # Create absorbance subfolder
        absorbance_folder = os.path.join(self.save_path, "absorbance")
        os.makedirs(absorbance_folder, exist_ok=True)
        
        name = self.ui.savepath.text() if self.ui and hasattr(self.ui, 'savepath') and self.ui.savepath.text() else ''
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        if name:
            filename = os.path.join(absorbance_folder, f"{name}_{timestamp}.txt")
        else:
            filename = os.path.join(absorbance_folder, f"{timestamp}.txt")
        
        try:
            with open(filename, 'w') as txtfile:
                txtfile.write(f'# GPS_LATITUDE: {self.current_gps["latitude"]:.6f}\n')
                txtfile.write(f'# GPS_LONGITUDE: {self.current_gps["longitude"]:.6f}\n')
                txtfile.write(f'# GPS_ALTITUDE: {self.current_gps["altitude"]:.2f}\n')
                txtfile.write('Wavelength (nm)\tAbsorbance\n')
                for i in range(len(self.lmbd)):
                    txtfile.write(f'{self.lmbd[i]:.2f}\t{self.absorbance_data[i]:.4f}\n')
            self.get_logger().info(f"Saved absorbance data to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save absorbance data: {e}")

    def set_integration_time(self):
        """Set integration time on spectrometer"""
        try:
            self.spec.integration_time_micros(self.integration_time_ms * 1000)
            self.get_logger().info(f"Integration time set to {self.integration_time_ms} ms.")
        except Exception as e:
            self.get_logger().error(f"Failed to set integration time: {e}")

    def joy_cb(self, msg):
        """Callback for joystick messages to save data and control functions"""
        """Callback for joystick messages to save data and control functions"""
        # X button - Save intensity data
        if len(msg.buttons) > self.joystick_intensity_button:
            if self.is_button_pressed(self.joystick_intensity_button, msg.buttons[self.joystick_intensity_button]):
                self.get_logger().info("X button pressed, saving intensity data...")
                self.save_intensity_data('txt')
        
        # Triangle button - Save reflectance data
        if len(msg.buttons) > self.joystick_reflectance_button:
            if self.is_button_pressed(self.joystick_reflectance_button, msg.buttons[self.joystick_reflectance_button]):
                self.get_logger().info("Triangle button pressed, saving all spectral data...")
                self.save_intensity_data('txt')
                self.save_reflectance_data('txt')
                self.save_absorbance_data('txt')
        
        # PS button - Toggle spectrometer connection
        if len(msg.buttons) > self.joystick_ps_button:
            if self.is_button_pressed(self.joystick_ps_button, msg.buttons[self.joystick_ps_button]):
                self.get_logger().info("PS button pressed, toggling spectrometer connection...")
                self.toggle_spectrometer_connection()
        
        # Start button - Toggle tab mode
        if len(msg.buttons) > self.joystick_start_button:
            if self.is_button_pressed(self.joystick_start_button, msg.buttons[self.joystick_start_button]):
                self.get_logger().info("Start button pressed, toggling tab mode...")
                self.toggle_tab_mode()
        
        # Square button - Toggle primary result service
        if len(msg.buttons) > self.joystick_square_button:
            if self.is_button_pressed(self.joystick_square_button, msg.buttons[self.joystick_square_button]):
                self.get_logger().info("Square button pressed, toggling primary result...")
                self.toggle_primary_result_service()
        
        # Circle button - Cycle threshold value
        if len(msg.buttons) > self.joystick_circle_button:
            if self.is_button_pressed(self.joystick_circle_button, msg.buttons[self.joystick_circle_button]):
                self.get_logger().info("Circle button pressed, cycling threshold...")
                self.cycle_threshold_value()

    def capture_averaged_spectrum(self):
        """Acquire and average multiple spectra"""
        accumulator = np.zeros_like(self.lmbd)
        for _ in range(self.scans_to_average):
            accumulator += self.spec.intensities()
            time.sleep(self.integration_time_ms / 1000.0)
        return accumulator / self.scans_to_average

    def calculate_reflectance(self, sample_spectrum):
        """Calculate reflectance and absorbance for UI display (basic calculation)"""
        if self.reference_spectrum is not None and self.matriks_spectrum is not None:
            # Apply boxcar smoothing to all spectra
            smoothed_sample = self.apply_boxcar(sample_spectrum)
            smoothed_ref = self.apply_boxcar(self.reference_spectrum)
            smoothed_matriks = self.apply_boxcar(self.matriks_spectrum)
            
            # Background values (either from file or default 804.761)
            if self.background_spectrum is not None:
                background_values = self.background_spectrum
            else:
                background_values = np.full_like(self.lmbd, 804.761)
            
            # Apply smoothing with window 300 after arithmetic operations
            from scipy.signal import savgol_filter
            
            numerator_before_smooth = (smoothed_sample - background_values) * smoothed_matriks
            denominator_before_smooth = (smoothed_ref - background_values) * smoothed_matriks
            
            # Ensure window size is appropriate
            window_size = min(300, len(numerator_before_smooth) // 3)
            if window_size % 2 == 0:
                window_size -= 1
            if window_size < 5:
                window_size = 5
                
            numerator = savgol_filter(numerator_before_smooth, window_size, 2)
            denominator = savgol_filter(denominator_before_smooth, window_size, 2)
            
            # Calculate reflectance (no multiplication by 100)
            safe_denominator = np.where(np.abs(denominator) > 1e-5, denominator, 1e-5)
            reflectance = numerator / safe_denominator
            
            # Store reflectance data
            self.reflectance_data = reflectance
            
            # Calculate absorbance: log10(1/(0.01*reflectance))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            absorbance = np.log10(1 / (0.01 * np.maximum(reflectance, epsilon)))
            
            # Store absorbance data
            self.absorbance_data = absorbance
            
            # Return reflectance for UI display
            return reflectance
        return None

    def apply_boxcar(self, spectrum):
        """Apply boxcar smoothing to spectrum"""
        if self.ui:
            width = self.ui.boxcar_width.value()
        else:
            width = self.boxcar_width
            
        if width <= 1:
            return spectrum
        
        kernel = np.ones(width) / width
        return np.convolve(spectrum, kernel, mode='same')

    def check_and_load_defaults(self):
        """Check for default reference, background, and matriks files and auto-enable reflectance mode"""
        # Look for files in the workspace root directory
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ref_path = os.path.join(workspace_root, "reference.txt")
        bg_path = os.path.join(workspace_root, "bg.txt")
        matriks_path = os.path.join(workspace_root, "matriks.txt")
        
        # Also check in current working directory as fallback
        if not os.path.exists(ref_path):
            ref_path = "reference.txt"
        if not os.path.exists(bg_path):
            bg_path = "bg.txt"
        if not os.path.exists(matriks_path):
            matriks_path = "matriks.txt"
        
        self.get_logger().info(f"Looking for reference file at: {ref_path}")
        self.get_logger().info(f"Looking for background file at: {bg_path}")
        self.get_logger().info(f"Looking for matriks file at: {matriks_path}")
        
        ref_loaded = False
        bg_loaded = False
        matriks_loaded = False
        
        # Try to load reference file
        if os.path.exists(ref_path):
            try:
                ref_loaded = self.load_spectrum_from_file(ref_path, "reference")
                if ref_loaded:
                    self.get_logger().info(f"Auto-loaded reference from {ref_path}")
            except Exception as e:
                self.get_logger().error(f"Error loading reference from {ref_path}: {e}")
        else:
            self.get_logger().warn(f"Reference file not found at {ref_path}")
        
        # Try to load background file, use fixed value if not found
        if os.path.exists(bg_path):
            try:
                bg_loaded = self.load_spectrum_from_file(bg_path, "background")
                if bg_loaded:
                    self.get_logger().info(f"Auto-loaded background from {bg_path}")
            except Exception as e:
                self.get_logger().error(f"Error loading background from {bg_path}: {e}")
        else:
            self.get_logger().info(f"Background file not found at {bg_path}")
        
        # If background file not found or failed to load, use fixed value
        if not bg_loaded:
            self.background_spectrum = np.full_like(self.lmbd, 804.761)
            bg_loaded = True
            self.get_logger().info("Using fixed background value: 804.761 for all wavelengths")
        
        # Try to load matriks file
        if os.path.exists(matriks_path):
            try:
                matriks_loaded = self.load_spectrum_from_file(matriks_path, "matriks")
                if matriks_loaded:
                    self.get_logger().info(f"Auto-loaded matriks from {matriks_path}")
            except Exception as e:
                self.get_logger().error(f"Error loading matriks from {matriks_path}: {e}")
        else:
            self.get_logger().warn(f"Matriks file not found at {matriks_path}")
    
        # Auto-enable reflectance mode if reference and matriks files loaded successfully
        # Background is always available (either from file or fixed value)
        if ref_loaded and bg_loaded and matriks_loaded and self.ui:
            self.ui.reflectance_mode.setChecked(True)
            self.reflectance_mode = True
            self.reflectance_step = 4  # Skip setup, go directly to measurement
            self.ui.reflectance_setup.setVisible(False)
            self.ui.tab_widget.setTabEnabled(1, True)
            self.ui.tab_widget.setCurrentIndex(1)  # Switch to reflectance tab
            self.get_logger().info("Auto-enabled reflectance mode with reference and matriks files")

    def load_spectrum_from_file(self, filepath, spectrum_type):
        """Load spectrum from text file"""
        try:
            wavelengths = []
            intensities = []
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                # Skip header line if it exists
                start_line = 1 if lines[0].lower().startswith('wavelength') else 0
                
                for line in lines[start_line:]:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Try different separators
                        if '\t' in line:
                            parts = line.split('\t')
                        elif ',' in line and len(line.split(',')) >= 2:
                            parts = line.split(',')
                        else:
                            parts = line.split()
                        
                        if len(parts) >= 2:
                            try:
                                # Handle European decimal format (comma as decimal separator)
                                wavelength_str = parts[0].strip().replace(',', '.')
                                intensity_str = parts[1].strip().replace(',', '.')
                                
                                wavelength = float(wavelength_str)
                                intensity = float(intensity_str)
                                
                                wavelengths.append(wavelength)
                                intensities.append(intensity)
                            except ValueError as e:
                                self.get_logger().debug(f"Skipping line: {line} - {e}")
                                continue
            
            if len(wavelengths) == 0:
                self.get_logger().error(f"No valid data found in {filepath}")
                return False
            
            self.get_logger().info(f"Loaded {len(wavelengths)} data points from {filepath}")
            self.get_logger().info(f"Wavelength range: {min(wavelengths):.2f} - {max(wavelengths):.2f} nm")
            self.get_logger().info(f"Spectrometer wavelength range: {self.lmbd.min():.2f} - {self.lmbd.max():.2f} nm")
            
            # Check if wavelength arrays have compatible ranges
            file_min, file_max = min(wavelengths), max(wavelengths)
            spec_min, spec_max = self.lmbd.min(), self.lmbd.max()
            
            if file_max < spec_min or file_min > spec_max:
                self.get_logger().error(f"Wavelength ranges don't overlap: file [{file_min:.2f}-{file_max:.2f}] vs spectrometer [{spec_min:.2f}-{spec_max:.2f}]")
                return False
            
            # Interpolate to match current wavelength array
            spectrum = np.interp(self.lmbd, wavelengths, intensities)
            
            if spectrum_type == "reference":
                self.reference_spectrum = spectrum
                self.get_logger().info(f"Reference spectrum loaded: min={spectrum.min():.2f}, max={spectrum.max():.2f}")
            elif spectrum_type == "background":
                self.background_spectrum = spectrum
                self.get_logger().info(f"Background spectrum loaded: min={spectrum.min():.2f}, max={spectrum.max():.2f}")
            elif spectrum_type == "matriks":
                self.matriks_spectrum = spectrum
                self.get_logger().info(f"Matriks spectrum loaded: min={spectrum.min():.2f}, max={spectrum.max():.2f}")
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error loading {spectrum_type} from {filepath}: {e}")
            return False

    def save_spectrum_to_file(self, spectrum, prefix):
        """Save spectrum to text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_path, f"{prefix}_{timestamp}.txt")
        try:
            with open(filename, 'w') as f:
                f.write("Wavelength (nm)\tIntensity\n")
                for w, i in zip(self.lmbd, spectrum):
                    f.write(f"{w:.2f}\t{i:.4f}\n")
            self.get_logger().info(f"Saved {prefix} spectrum to {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save {prefix} spectrum: {e}")

    def destroy_node(self):
        """Clean shutdown"""
        self.acquisition_running = False
        if hasattr(self, 'acquisition_thread') and self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        if hasattr(self, 'spec'):
            try:
                self.spec.close()
            except:
                pass
        super().destroy_node()


def main(args=None):
    import signal
    import sys
    
    rclpy.init(args=args)
    
    # Check if QApplication already exists
    app = None
    try:
        if not qt_app.instance():
            app = qt_app([])
        else:
            app = qt_app.instance()
    except:
        if not QtGui.QApplication.instance():
            app = QtGui.QApplication([])
        else:
            app = QtGui.QApplication.instance()
    
    node = SpectroNode()
    
    # Setup signal handlers for proper shutdown
    def signal_handler(signum, frame):
        print("\nShutdown signal received, closing...")
        try:
            node.destroy_node()
            rclpy.shutdown()
            if app:
                app.quit()
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    def spin_node():
        """Spin ROS node in separate thread"""
        try:
            rclpy.spin(node)
        except Exception as e:
            print(f"ROS spin error: {e}")
        finally:
            try:
                node.destroy_node()
            except:
                pass
    
    # Start ROS spinning in background thread
    ros_thread = threading.Thread(target=spin_node, daemon=True)
    ros_thread.start()
    
    try:
        if node.ui and app:
            # Run Qt event loop if UI is available
            app.exec_()
        else:
            # Just keep the node running without UI
            while rclpy.ok():
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received, shutting down...")
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
