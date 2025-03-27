import os
import sys

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

# PyQt imports
from PyQt5.QtCore import Qt, QSize, QTimer, QPoint, QRect, pyqtSignal, QEvent, QSettings
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPalette, QColor, QCursor, QPainter, QWheelEvent, QTouchEvent, QFont
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QGridLayout,
    QDesktopWidget,
    QFontDialog,
)
# Local application imports
from functions import *

# Custom classes for touchpad/mouse wheel gesture support
class CustomScrollArea(QScrollArea):
    """Custom scroll area with mouse wheel zoom support and touchpad gestures"""
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    pan_requested = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.last_pos = QPoint()
        self.panning = False
        
        # For touchpad gesture tracking
        self.last_touch_pos = QPoint()
        self.touch_panning = False
        
        # Enable viewport to receive touch events directly
        self.viewport().setAttribute(Qt.WA_AcceptTouchEvents, True)
    
    def viewportEvent(self, event):
        """Handle viewport events including touch events"""
        if event.type() == QEvent.TouchBegin:
            # Start of touch - may be for panning or other gestures
            touch_points = event.touchPoints()
            if len(touch_points) == 2:  # Two-finger touch - likely panning
                self.touch_panning = True
                self.last_touch_pos = touch_points[0].pos() + touch_points[1].pos() / 2  # Average position
                return True
            
        elif event.type() == QEvent.TouchUpdate and self.touch_panning:
            # Update during touch - move/pan the view
            touch_points = event.touchPoints()
            if len(touch_points) == 2:
                # Calculate center point between two fingers
                current_pos = touch_points[0].pos() + touch_points[1].pos() / 2
                
                # Calculate how much to pan
                delta = current_pos - self.last_touch_pos
                
                # Emit signal to handle panning
                self.pan_requested.emit(delta)
                
                # Update last position
                self.last_touch_pos = current_pos
                return True
                
        elif event.type() == QEvent.TouchEnd:
            # End of touch
            self.touch_panning = False
        
        # For non-touch events or events we don't handle, use default behavior
        return super().viewportEvent(event)
    
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Zoom with Ctrl+wheel
            if event.angleDelta().y() > 0:
                self.zoom_in_requested.emit()
            else:
                self.zoom_out_requested.emit()
            event.accept()
        elif event.modifiers() & Qt.ShiftModifier:
            # Horizontal scroll with Shift+wheel
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - event.angleDelta().y()
            )
            event.accept()
        else:
            # Default vertical scroll behavior
            super().wheelEvent(event)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and event.modifiers() & Qt.ControlModifier):
            self.panning = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.pos() - self.last_pos
            self.pan_requested.emit(delta)
            self.last_pos = event.pos()
            event.accept()
        else:
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self.panning and (event.button() == Qt.MiddleButton or event.button() == Qt.LeftButton):
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)


class CustomImageLabel(QLabel):
    """Custom label with touchpad gesture support"""
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        
        # Specifically for pinch to zoom
        self.setAttribute(Qt.WA_AcceptTouchEvents, True)
        
        # Track pinch distance
        self.last_pinch_distance = 0
    
    def event(self, event):
        if event.type() == QEvent.TouchBegin:
            # Start tracking for pinch zoom
            touch_points = event.touchPoints()
            if len(touch_points) == 2:
                p1 = touch_points[0].pos()
                p2 = touch_points[1].pos()
                self.last_pinch_distance = (p1 - p2).manhattanLength()
            return True
            
        elif event.type() == QEvent.TouchUpdate:
            # Check for pinch zoom
            touch_points = event.touchPoints()
            if len(touch_points) == 2:
                p1 = touch_points[0].pos()
                p2 = touch_points[1].pos()
                current_distance = (p1 - p2).manhattanLength()
                
                # Only emit if change is significant
                if self.last_pinch_distance > 0:
                    diff = current_distance - self.last_pinch_distance
                    if diff > 20:  # Zoom in - fingers moving apart
                        self.zoom_in_requested.emit()
                        self.last_pinch_distance = current_distance
                        return True
                    elif diff < -20:  # Zoom out - fingers moving together
                        self.zoom_out_requested.emit()
                        self.last_pinch_distance = current_distance
                        return True
                
                self.last_pinch_distance = current_distance
            return True
            
        elif event.type() == QEvent.TouchEnd:
            self.last_pinch_distance = 0
            return True
            
        return super().event(event)
    
    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            # Zoom with Ctrl+wheel
            if event.angleDelta().y() > 0:
                self.zoom_in_requested.emit()
            else:
                self.zoom_out_requested.emit()
            event.accept()
        else:
            # Pass the event to parent
            super().wheelEvent(event)


class PhotoEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize variables
        self.original_image = None
        self.current_image = None
        self.file_path = None
        
        # Initialize history stack for undo
        self.history = []
        self.max_history = 10  # Limit history stack to 10 steps
        
        # For previewing changes
        self.preview_image = None
        self.preview_timer = None
        
        # For zoom and pan
        self.scale_factor = 1.0
        
        # For comparison view
        self.comparison_mode = False
        
        # For font size scaling
        self.font_size = 9  # Default font size
        self.settings = QSettings("PhotoEditor", "FontSettings")
        self.load_settings()
        
        # Set up the UI
        self.initUI()
        
    def load_settings(self):
        """Load application settings from QSettings"""
        self.font_size = self.settings.value("font_size", 9, type=int)
        
    def save_settings(self):
        """Save application settings to QSettings"""
        self.settings.setValue("font_size", self.font_size)
        
    def update_font_size_from_menu(self, size):
        """Update font size from menu selection and update checked state"""
        # Update the font size
        self.change_font_size(size)
        
        # Update the combo box selection
        self.font_size_combobox.setCurrentIndex(self.font_size_combobox.findText(f"{size}pt"))
        
        # Update checked state in menu items
        for s, action in self.font_size_actions.items():
            action.setChecked(s == size)
    
    def initUI(self):
        # Get screen dimensions
        screen_rect = QDesktopWidget().availableGeometry()
        screen_width = screen_rect.width()
        screen_height = screen_rect.height()
        
        # Set window properties with relative sizing
        self.setWindowTitle("Photo Editor")
        
        # Set window size to 80% of screen size
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.setGeometry(
            (screen_width - window_width) // 2,  # Center horizontally
            (screen_height - window_height) // 2,  # Center vertically
            window_width,
            window_height
        )
        
        # Set up central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create splitter for resizable sections
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Set up image display area
        self.setup_image_display()
        
        # Set up control panel
        self.setup_control_panel()
        
        # Add the splitter to the main layout
        self.main_layout.addWidget(self.splitter)
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress indicator
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMaximumWidth(150)
        self.status_bar.addPermanentWidget(self.progress)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.create_toolbar()
        
        # Setup tooltips
        self.setup_tooltips()
        
    def setup_image_display(self):
        # Create a widget for the image display
        self.image_widget = QWidget()
        self.image_layout = QVBoxLayout(self.image_widget)
        
        # Create container for the image view
        self.image_container = QWidget()
        self.image_container_layout = QHBoxLayout(self.image_container)
        self.image_container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create a label to display the current image
        self.image_label = CustomImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setText("Open an image to begin editing")
        self.image_label.setStyleSheet("background-color: #252525; padding: 10px;")
        self.image_label.setMinimumSize(300, 200)
        
        # Create a label to display the original image (for comparison)
        self.original_image_label = QLabel()
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("background-color: #252525; padding: 10px;")
        self.original_image_label.setMinimumSize(300, 200)
        self.original_image_label.hide()  # Hidden by default
        
        # Add both labels to the container
        self.image_container_layout.addWidget(self.original_image_label)
        self.image_container_layout.addWidget(self.image_label)
        
        # Create a scroll area for the image container
        self.scroll_area = CustomScrollArea()
        self.scroll_area.setWidget(self.image_container)
        self.scroll_area.setWidgetResizable(True)
        
        # Enable touch support for scroll area
        self.scroll_area.viewport().setAttribute(Qt.WA_AcceptTouchEvents)
        
        # Connect the scroll area signals to our zoom handlers
        self.scroll_area.zoom_in_requested.connect(self.zoom_in)
        self.scroll_area.zoom_out_requested.connect(self.zoom_out)
        self.scroll_area.pan_requested.connect(self.handle_pan)
        
        # Also connect image label signals
        self.image_label.zoom_in_requested.connect(self.zoom_in)
        self.image_label.zoom_out_requested.connect(self.zoom_out)
        
        # Create zoom controls
        self.zoom_layout = QHBoxLayout()
        
        self.zoom_out_btn = QPushButton("Zoom Out")
        self.zoom_out_btn.setToolTip("Zoom Out (Ctrl+-)")
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        
        self.zoom_reset_btn = QPushButton("Center")
        self.zoom_reset_btn.setToolTip("Reset Zoom (Ctrl+0)")
        self.zoom_reset_btn.clicked.connect(self.zoom_reset)
        
        self.zoom_in_btn = QPushButton("Zoom In")
        self.zoom_in_btn.setToolTip("Zoom In (Ctrl+=)")
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        
        self.comparison_toggle = QCheckBox("Side-by-side Comparison")
        self.comparison_toggle.setToolTip("Show original image for comparison (Ctrl+D)")
        self.comparison_toggle.toggled.connect(self.toggle_comparison)
        
        self.zoom_layout.addWidget(self.zoom_out_btn)
        self.zoom_layout.addWidget(self.zoom_reset_btn)
        self.zoom_layout.addWidget(self.zoom_in_btn)
        self.zoom_layout.addStretch()
        self.zoom_layout.addWidget(self.comparison_toggle)
        
        # Add the scroll area and zoom controls to the image layout
        self.image_layout.addWidget(self.scroll_area)
        self.image_layout.addLayout(self.zoom_layout)
        
        # Add the image widget to the splitter
        self.splitter.addWidget(self.image_widget)
        
    def setup_control_panel(self):
        # Create a widget for the control panel
        self.control_widget = QWidget()
        self.control_layout = QVBoxLayout(self.control_widget)
        
        # Create tabs for different categories of adjustments
        self.control_tabs = QTabWidget()
        
        # Create tabs for different functionality
        self.create_basic_adjustments_tab()
        self.create_filters_tab()
        self.create_analyze_tab()
        
        # Add the tabs to the control layout
        self.control_layout.addWidget(self.control_tabs)
        
        # Add buttons for common operations
        self.create_common_buttons()
        
        # Add the control widget to the splitter
        self.splitter.addWidget(self.control_widget)
        
        # Set the initial sizes of the splitter
        self.splitter.setSizes([800, 400])
        
    def select_font_dialog(self):
        """Open font selection dialog"""
        current_font = QApplication.font()
        font, ok = QFontDialog.getFont(current_font, self, "Select Font")
        if ok:
            # Apply the font to the entire application
            self.font_size = font.pointSize()
            QApplication.setFont(font)
            
            # Re-apply styles with new font size to ensure consistency
            apply_style(QApplication.instance(), self.font_size)
            
            self.status_bar.showMessage(f"Font changed to {font.family()}, {font.pointSize()}pt")
            self.save_settings()
        
    def change_font_size(self, size):
        """Change the application font size"""
        self.font_size = size
        
        # Apply new font size to application
        font = QApplication.font()
        font.setPointSize(self.font_size)
        QApplication.setFont(font)
        
        # Re-apply styles with new font size
        apply_style(QApplication.instance(), self.font_size)
        
        self.save_settings()
        
    def create_basic_adjustments_tab(self):
        # Create the basic adjustments tab
        self.basic_tab = QWidget()
        self.basic_layout = QVBoxLayout(self.basic_tab)
        
        # Grayscale group
        self.grayscale_group = QGroupBox("Grayscale Conversion")
        self.grayscale_layout = QVBoxLayout()
        
        # Grayscale methods
        self.grayscale_combo = QComboBox()
        self.grayscale_combo.addItems(["Luminance", "Lightness", "Average"])
        
        self.grayscale_button = QPushButton("Apply Grayscale")
        self.grayscale_button.clicked.connect(self.apply_grayscale)
        
        self.grayscale_layout.addWidget(QLabel("Method:"))
        self.grayscale_layout.addWidget(self.grayscale_combo)
        self.grayscale_layout.addWidget(self.grayscale_button)
        self.grayscale_group.setLayout(self.grayscale_layout)
        
        # Brightness group
        self.brightness_group = QGroupBox("Brightness")
        self.brightness_layout = QVBoxLayout()
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-255)
        self.brightness_slider.setMaximum(255)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(50)
        self.brightness_slider.valueChanged.connect(self.preview_brightness)
        
        self.brightness_value = QLabel("0")
        self.brightness_slider.valueChanged.connect(lambda: self.brightness_value.setText(str(self.brightness_slider.value())))
        
        self.brightness_apply = QPushButton("Apply Brightness")
        self.brightness_apply.clicked.connect(self.apply_brightness)
        
        self.brightness_layout.addWidget(self.brightness_slider)
        self.brightness_layout.addWidget(self.brightness_value)
        self.brightness_layout.addWidget(self.brightness_apply)
        self.brightness_group.setLayout(self.brightness_layout)
        
        # Contrast group
        self.contrast_group = QGroupBox("Contrast")
        self.contrast_layout = QVBoxLayout()
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(-127)
        self.contrast_slider.setMaximum(127)
        self.contrast_slider.setValue(0)
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(25)
        self.contrast_slider.valueChanged.connect(self.preview_contrast)
        
        self.contrast_value = QLabel("0")
        self.contrast_slider.valueChanged.connect(lambda: self.contrast_value.setText(str(self.contrast_slider.value())))
        
        self.contrast_apply = QPushButton("Apply Contrast")
        self.contrast_apply.clicked.connect(self.apply_contrast)
        
        self.contrast_layout.addWidget(self.contrast_slider)
        self.contrast_layout.addWidget(self.contrast_value)
        self.contrast_layout.addWidget(self.contrast_apply)
        self.contrast_group.setLayout(self.contrast_layout)
        
        # Inversion group
        self.inversion_group = QGroupBox("Invert Colors")
        self.inversion_layout = QVBoxLayout()
        
        self.inversion_button = QPushButton("Invert Image")
        self.inversion_button.clicked.connect(self.apply_inversion)
        
        self.inversion_layout.addWidget(self.inversion_button)
        self.inversion_group.setLayout(self.inversion_layout)
        
        # Binarization group
        self.binary_group = QGroupBox("Binarization")
        self.binary_layout = QVBoxLayout()
        
        self.binary_slider = QSlider(Qt.Horizontal)
        self.binary_slider.setMinimum(0)
        self.binary_slider.setMaximum(255)
        self.binary_slider.setValue(127)
        self.binary_slider.setTickPosition(QSlider.TicksBelow)
        self.binary_slider.setTickInterval(25)
        self.binary_slider.valueChanged.connect(self.preview_binarization)
        
        self.binary_value = QLabel("127")
        self.binary_slider.valueChanged.connect(lambda: self.binary_value.setText(str(self.binary_slider.value())))
        
        self.binary_apply = QPushButton("Apply Binarization")
        self.binary_apply.clicked.connect(self.apply_binarization)
        
        self.binary_layout.addWidget(self.binary_slider)
        self.binary_layout.addWidget(self.binary_value)
        self.binary_layout.addWidget(self.binary_apply)
        self.binary_group.setLayout(self.binary_layout)
        
        # Add all groups to the basic layout
        self.basic_layout.addWidget(self.grayscale_group)
        self.basic_layout.addWidget(self.brightness_group)
        self.basic_layout.addWidget(self.contrast_group)
        self.basic_layout.addWidget(self.inversion_group)
        self.basic_layout.addWidget(self.binary_group)
        self.basic_layout.addStretch()
        
        # Add the basic tab to the control tabs
        self.control_tabs.addTab(self.basic_tab, "Basic Adjustments")
        
    def create_filters_tab(self):
        # Create the filters tab with scrollable area
        self.filters_tab = QWidget()
        
        # Create a scroll area to make filters scrollable
        self.filters_scroll_area = QScrollArea()
        self.filters_scroll_area.setWidgetResizable(True)
        self.filters_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Create content widget for the scroll area
        self.filters_content = QWidget()
        self.filters_layout = QVBoxLayout(self.filters_content)
        
        # Sharpness group
        self.sharp_group = QGroupBox("Sharpness")
        self.sharp_layout = QVBoxLayout()
        
        self.sharp_slider = QDoubleSpinBox()
        self.sharp_slider.setMinimum(0.1)
        self.sharp_slider.setMaximum(5.0)
        self.sharp_slider.setValue(1.0)
        self.sharp_slider.setSingleStep(0.1)
        self.sharp_slider.valueChanged.connect(self.preview_sharpness)
        
        self.sharp_apply = QPushButton("Apply Sharpness")
        self.sharp_apply.clicked.connect(self.apply_sharpness)
        
        self.sharp_layout.addWidget(QLabel("Strength:"))
        self.sharp_layout.addWidget(self.sharp_slider)
        self.sharp_layout.addWidget(self.sharp_apply)
        self.sharp_group.setLayout(self.sharp_layout)
        
        # Mean filter group
        self.mean_group = QGroupBox("Mean Filter (Box Blur)")
        self.mean_layout = QVBoxLayout()
        
        self.mean_kernel = QSpinBox()
        self.mean_kernel.setMinimum(3)
        self.mean_kernel.setMaximum(25)
        self.mean_kernel.setValue(3)
        self.mean_kernel.setSingleStep(2)  # Only odd values
        
        self.mean_apply = QPushButton("Apply Mean Filter")
        self.mean_apply.clicked.connect(self.apply_mean_filter)
        
        self.mean_layout.addWidget(QLabel("Kernel Size:"))
        self.mean_layout.addWidget(self.mean_kernel)
        self.mean_layout.addWidget(self.mean_apply)
        self.mean_group.setLayout(self.mean_layout)
        
        # Gaussian blur group
        self.gauss_group = QGroupBox("Gaussian Blur")
        self.gauss_layout = QVBoxLayout()
        
        self.gauss_kernel = QSpinBox()
        self.gauss_kernel.setMinimum(1)
        self.gauss_kernel.setMaximum(25)
        self.gauss_kernel.setValue(3)
        
        self.gauss_sigma = QDoubleSpinBox()
        self.gauss_sigma.setMinimum(0.1)
        self.gauss_sigma.setMaximum(10.0)
        self.gauss_sigma.setValue(1.0)
        self.gauss_sigma.setSingleStep(0.1)
        
        self.gauss_apply = QPushButton("Apply Gaussian Blur")
        self.gauss_apply.clicked.connect(self.apply_gaussian_blur)
        
        self.gauss_layout.addWidget(QLabel("Kernel Size:"))
        self.gauss_layout.addWidget(self.gauss_kernel)
        self.gauss_layout.addWidget(QLabel("Sigma:"))
        self.gauss_layout.addWidget(self.gauss_sigma)
        self.gauss_layout.addWidget(self.gauss_apply)
        self.gauss_group.setLayout(self.gauss_layout)
        
        # NEW: Median filter group
        self.median_group = QGroupBox("Median Filter")
        self.median_layout = QVBoxLayout()
        
        self.median_kernel = QSpinBox()
        self.median_kernel.setMinimum(3)
        self.median_kernel.setMaximum(25)
        self.median_kernel.setValue(3)
        self.median_kernel.setSingleStep(2)  # Only odd values
        
        self.median_apply = QPushButton("Apply Median Filter")
        self.median_apply.clicked.connect(self.apply_median_filter)
        
        self.median_layout.addWidget(QLabel("Kernel Size:"))
        self.median_layout.addWidget(self.median_kernel)
        self.median_layout.addWidget(self.median_apply)
        self.median_group.setLayout(self.median_layout)
        
        # NEW: Edge detection group
        self.edge_group = QGroupBox("Edge Detection")
        self.edge_layout = QVBoxLayout()
        
        # Edge detection method selection
        self.edge_method_label = QLabel("Method:")
        self.edge_method_combo = QComboBox()
        self.edge_method_combo.addItems(["Roberts Cross", "Prewitt", "Prewitt (All Directions)", 
                                         "Prewitt (Gradient Magnitude)", "Sobel", "Sobel (Gradient Magnitude)"])
        self.edge_method_combo.currentIndexChanged.connect(self.update_edge_controls)
        
        # Direction controls for Prewitt
        self.prewitt_direction_label = QLabel("Direction:")
        self.prewitt_direction_combo = QComboBox()
        self.prewitt_direction_combo.addItems(["0: North (top)", "1: North-East (top-right)", 
                                              "2: East (right)", "3: South-East (bottom-right)",
                                              "4: South (bottom)", "5: South-West (bottom-left)",
                                              "6: West (left)", "7: North-West (top-left)"])
        
        # Direction controls for Sobel
        self.sobel_direction_label = QLabel("Direction (degrees):")
        self.sobel_direction_combo = QComboBox()
        self.sobel_direction_combo.addItems(["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"])
        
        # Apply button
        self.edge_apply = QPushButton("Apply Edge Detection")
        self.edge_apply.clicked.connect(self.apply_edge_detection)
        
        # Add widgets to the layout
        self.edge_layout.addWidget(self.edge_method_label)
        self.edge_layout.addWidget(self.edge_method_combo)
        self.edge_layout.addWidget(self.prewitt_direction_label)
        self.edge_layout.addWidget(self.prewitt_direction_combo)
        self.edge_layout.addWidget(self.sobel_direction_label)
        self.edge_layout.addWidget(self.sobel_direction_combo)
        self.edge_layout.addWidget(self.edge_apply)
        self.edge_group.setLayout(self.edge_layout)
        
        # Initially hide direction controls - will show based on selected method
        self.update_edge_controls()
        
        # Custom kernel group
        self.custom_kernel_group = QGroupBox("Custom Kernel")
        self.custom_kernel_layout = QVBoxLayout()

        # Kernel size selection
        self.kernel_size_layout = QHBoxLayout()
        self.kernel_size_label = QLabel("Kernel Size:")
        self.kernel_size_combo = QComboBox()
        self.kernel_size_combo.addItems(["3x3", "5x5", "7x7"])
        self.kernel_size_combo.currentIndexChanged.connect(self.update_kernel_grid)

        self.kernel_size_layout.addWidget(self.kernel_size_label)
        self.kernel_size_layout.addWidget(self.kernel_size_combo)

        # Kernel grid for input values
        self.kernel_grid_widget = QWidget()
        self.kernel_grid_layout = QGridLayout(self.kernel_grid_widget)

        # Normalization option
        self.normalize_kernel_check = QCheckBox("Normalize Kernel")
        self.normalize_kernel_check.setChecked(True)
        self.normalize_kernel_check.setToolTip("Automatically normalize the kernel values to prevent extreme brightness/darkness")

        # Predefined kernels dropdown
        self.predefined_layout = QHBoxLayout()
        self.predefined_label = QLabel("Predefined:")
        self.predefined_combo = QComboBox()
        self.predefined_combo.addItems(["Custom", "Blur", "Sharpen", "Edge Detection", "Emboss"])
        self.predefined_combo.currentIndexChanged.connect(self.apply_predefined_kernel)

        self.predefined_layout.addWidget(self.predefined_label)
        self.predefined_layout.addWidget(self.predefined_combo)

        # Apply button
        self.custom_kernel_apply = QPushButton("Apply Custom Kernel")
        self.custom_kernel_apply.clicked.connect(self.apply_custom_kernel)

        # Add all to the layout
        self.custom_kernel_layout.addLayout(self.kernel_size_layout)
        self.custom_kernel_layout.addWidget(self.kernel_grid_widget)
        self.custom_kernel_layout.addWidget(self.normalize_kernel_check)
        self.custom_kernel_layout.addLayout(self.predefined_layout)
        self.custom_kernel_layout.addWidget(self.custom_kernel_apply)
        self.custom_kernel_group.setLayout(self.custom_kernel_layout)
        
        # Add all filter groups to the layout
        self.filters_layout.addWidget(self.sharp_group)
        self.filters_layout.addWidget(self.mean_group)
        self.filters_layout.addWidget(self.gauss_group)
        # NEW: Add median filter group
        self.filters_layout.addWidget(self.median_group)
        # NEW: Add edge detection group
        self.filters_layout.addWidget(self.edge_group)
        self.filters_layout.addWidget(self.custom_kernel_group)
        self.filters_layout.addStretch()
        
        # Set the scroll area's widget to the content widget
        self.filters_scroll_area.setWidget(self.filters_content)
        
        # Add the scroll area to the filters tab
        filters_main_layout = QVBoxLayout(self.filters_tab)
        filters_main_layout.addWidget(self.filters_scroll_area)
        filters_main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add the filters tab to the control tabs
        self.control_tabs.addTab(self.filters_tab, "Filters")
        
        # Initialize the 3x3 kernel grid
        self.update_kernel_grid()
        
    def create_analyze_tab(self):
        # Create the analyze tab
        self.analyze_tab = QWidget()
        self.analyze_layout = QVBoxLayout(self.analyze_tab)
        
        # Histogram button
        self.histogram_button = QPushButton("Show Histogram")
        self.histogram_button.clicked.connect(self.show_histogram)
        
        # Projections button
        self.projections_button = QPushButton("Show Projections")
        self.projections_button.clicked.connect(self.show_projections)
        
        # Add buttons to the analyze layout
        self.analyze_layout.addWidget(self.histogram_button)
        self.analyze_layout.addWidget(self.projections_button)
        self.analyze_layout.addStretch()
        
        # Add the analyze tab to the control tabs
        self.control_tabs.addTab(self.analyze_tab, "Analyze")
        
    def create_common_buttons(self):
        # Create a layout for common buttons
        self.buttons_layout = QHBoxLayout()
        
        # Reset button
        self.reset_button = QPushButton("Reset to Original")
        self.reset_button.clicked.connect(self.reset_image)
        self.reset_button.setEnabled(False)
        
        # Undo button 
        self.undo_button = QPushButton("Undo")
        self.undo_button.clicked.connect(self.undo_last_change)
        self.undo_button.setEnabled(False)
        
        # Add buttons to the layout
        self.buttons_layout.addWidget(self.reset_button)
        self.buttons_layout.addWidget(self.undo_button)
        
        # Add the buttons layout to the control layout
        self.control_layout.addLayout(self.buttons_layout)
        
    def create_menu_bar(self):
        # Create menu bar
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Open action
        open_action = QAction(self.style().standardIcon(self.style().SP_DialogOpenButton), "Open", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_image)
        file_menu.addAction(open_action)
        
        # Save action
        save_action = QAction(self.style().standardIcon(self.style().SP_DialogSaveButton), "Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_image)
        file_menu.addAction(save_action)
        
        # Save As action
        save_as_action = QAction(self.style().standardIcon(self.style().SP_DriveFDIcon), "Save As", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_image_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction(self.style().standardIcon(self.style().SP_DialogCloseButton), "Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        # Reset action
        reset_action = QAction(self.style().standardIcon(self.style().SP_BrowserReload), "Reset to Original", self)
        reset_action.setShortcut("Ctrl+R")
        reset_action.triggered.connect(self.reset_image)
        edit_menu.addAction(reset_action)
        
        # Undo action 
        undo_action = QAction(self.style().standardIcon(self.style().SP_ArrowBack), "Undo", self)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.triggered.connect(self.undo_last_change)
        edit_menu.addAction(undo_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        # Zoom In action
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl+=")  
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)
        
        # Zoom Out action
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)
        
        # Reset Zoom action
        zoom_reset_action = QAction("Reset Zoom", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self.zoom_reset)
        view_menu.addAction(zoom_reset_action)
        
        view_menu.addSeparator()
        
        # Side-by-side comparison action
        comparison_action = QAction("Toggle Side-by-side Comparison", self)
        comparison_action.setShortcut("Ctrl+D")
        comparison_action.triggered.connect(lambda: self.comparison_toggle.toggle())
        view_menu.addAction(comparison_action)
        
        # Add Font Size submenu
        font_menu = view_menu.addMenu("Font Size")
        
        # Font size actions
        self.font_size_actions = {}
        font_sizes = [7, 8, 9, 10, 11, 12, 14, 16]
        for size in font_sizes:
            size_action = QAction(f"{size}pt", self)
            size_action.setCheckable(True)
            if size == self.font_size:
                size_action.setChecked(True)
            size_action.triggered.connect(lambda checked, s=size: self.update_font_size_from_menu(s))
            font_menu.addAction(size_action)
            self.font_size_actions[size] = size_action
            
        # Custom font action
        custom_font_action = QAction("Custom Font...", self)
        custom_font_action.triggered.connect(self.select_font_dialog)
        font_menu.addAction(custom_font_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction(self.style().standardIcon(self.style().SP_MessageBoxInformation), "About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
    def create_toolbar(self):
        # Create toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        
        # Open action
        open_action = QAction(self.style().standardIcon(self.style().SP_DialogOpenButton), "Open", self)
        open_action.triggered.connect(self.open_image)
        toolbar.addAction(open_action)
        
        # Save action
        save_action = QAction(self.style().standardIcon(self.style().SP_DialogSaveButton), "Save", self)
        save_action.triggered.connect(self.save_image)
        toolbar.addAction(save_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Reset action
        reset_action = QAction(self.style().standardIcon(self.style().SP_BrowserReload), "Reset", self)
        reset_action.triggered.connect(self.reset_image)
        toolbar.addAction(reset_action)
        
        # Undo action 
        undo_action = QAction(self.style().standardIcon(self.style().SP_ArrowBack), "Undo", self)
        undo_action.triggered.connect(self.undo_last_change)
        toolbar.addAction(undo_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Zoom controls
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        toolbar.addAction(zoom_in_action)
        
        zoom_reset_action = QAction("Center", self)
        zoom_reset_action.triggered.connect(self.zoom_reset)
        toolbar.addAction(zoom_reset_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        toolbar.addAction(zoom_out_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Side-by-side comparison
        comparison_action = QAction(self.style().standardIcon(self.style().SP_FileDialogDetailedView), "Toggle Comparison", self)
        comparison_action.triggered.connect(lambda: self.comparison_toggle.toggle())
        toolbar.addAction(comparison_action)
        
        # Add separator
        toolbar.addSeparator()
        
        # Font size controls in toolbar
        font_size_label = QLabel("Font Size: ")
        toolbar.addWidget(font_size_label)
        
        self.font_size_combobox = QComboBox()
        self.font_size_combobox.addItems(["7pt", "8pt", "9pt", "10pt", "11pt", "12pt", "14pt", "16pt"])
        self.font_size_combobox.setCurrentIndex(self.font_size_combobox.findText(f"{self.font_size}pt"))
        self.font_size_combobox.currentIndexChanged.connect(
            lambda index: self.change_font_size(int(self.font_size_combobox.currentText().replace("pt", "")))
        )
        toolbar.addWidget(self.font_size_combobox)
        
        # Custom font button
        font_button = QAction(self.style().standardIcon(self.style().SP_DirHomeIcon), "Select Font", self)
        font_button.triggered.connect(self.select_font_dialog)
        toolbar.addAction(font_button)
    
    def setup_tooltips(self):
        # Basic adjustments tab
        self.grayscale_combo.setToolTip("Choose the method for converting to grayscale")
        self.grayscale_button.setToolTip("Convert the image to grayscale using the selected method")
        
        self.brightness_slider.setToolTip("Adjust the brightness of the image")
        self.brightness_apply.setToolTip("Apply the brightness adjustment")
        
        self.contrast_slider.setToolTip("Adjust the contrast of the image")
        self.contrast_apply.setToolTip("Apply the contrast adjustment")
        
        self.inversion_button.setToolTip("Invert the colors of the image")
        
        self.binary_slider.setToolTip("Set the threshold for binarization")
        self.binary_apply.setToolTip("Convert the image to binary (black and white) using the threshold")
        
        # Filters tab
        self.sharp_slider.setToolTip("Adjust the strength of the sharpening filter")
        self.sharp_apply.setToolTip("Apply the sharpening filter")
        
        self.mean_kernel.setToolTip("Set the size of the kernel for the mean filter")
        self.mean_apply.setToolTip("Apply the mean filter (box blur)")
        
        self.gauss_kernel.setToolTip("Set the size of the kernel for Gaussian blur")
        self.gauss_sigma.setToolTip("Set the sigma value for Gaussian blur")
        self.gauss_apply.setToolTip("Apply the Gaussian blur filter")
        
        # NEW: Tooltips for median filter
        self.median_kernel.setToolTip("Set the size of the kernel for the median filter")
        self.median_apply.setToolTip("Apply the median filter to reduce noise while preserving edges")
        
        # NEW: Tooltips for edge detection
        self.edge_method_combo.setToolTip("Select the edge detection method to use")
        self.prewitt_direction_combo.setToolTip("Select the direction for Prewitt edge detection")
        self.sobel_direction_combo.setToolTip("Select the direction (in degrees) for Sobel edge detection")
        self.edge_apply.setToolTip("Apply the selected edge detection method")
        
        # Custom kernel tooltips
        self.kernel_size_combo.setToolTip("Select the size of the custom kernel")
        self.custom_kernel_apply.setToolTip("Apply the custom kernel to the image")
        self.predefined_combo.setToolTip("Select a predefined kernel or create a custom one")
        
        # Analyze tab
        self.histogram_button.setToolTip("Show the histogram of the image")
        self.projections_button.setToolTip("Show the horizontal and vertical projections of the image")
        
        # Common buttons
        self.reset_button.setToolTip("Reset the image to its original state")
        self.undo_button.setToolTip("Undo the last change")
        
        # Font size tooltip in toolbar
        self.font_size_combobox.setToolTip("Change the application font size")
        
    def closeEvent(self, event):
        """Save settings when the application closes"""
        self.save_settings()
        event.accept()

    # NEW: Update edge detection controls based on selected method
    def update_edge_controls(self):
        selected_method = self.edge_method_combo.currentText()
        
        # Hide all direction controls by default
        self.prewitt_direction_label.hide()
        self.prewitt_direction_combo.hide()
        self.sobel_direction_label.hide()
        self.sobel_direction_combo.hide()
        
        # Show relevant controls based on selected method
        if selected_method == "Prewitt":
            self.prewitt_direction_label.show()
            self.prewitt_direction_combo.show()
        elif selected_method == "Sobel":
            self.sobel_direction_label.show()
            self.sobel_direction_combo.show()
            
    # Function implementations
    def open_image(self):
        # Open a file dialog to select an image
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", 
                                                 "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)", 
                                                 options=options)
        
        if file_path:
            try:
                # Show progress
                self.start_progress()
                
                # Read the image using OpenCV
                self.file_path = file_path
                self.original_image = cv2.imread(file_path)
                self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                # Initialize the current image
                self.current_image = self.original_image.copy()
                
                # Reset zoom factor for new image
                self.scale_factor = 1.0
                
                # Update the displayed image
                self.update_image_display()
                
                # Enable buttons
                self.reset_button.setEnabled(True)
                
                # Clear history
                self.history = []
                self.undo_button.setEnabled(False)
                
                # Update status bar
                self.status_bar.showMessage(f"Opened: {os.path.basename(file_path)}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not open the image: {str(e)}")
    
    def update_image_display(self):
        if self.current_image is not None:
            # Convert the numpy array to QImage
            height, width, channel = self.current_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Create a pixmap from the QImage
            current_pixmap = QPixmap.fromImage(q_img)
            
            # Apply zoom if needed
            if self.scale_factor != 1.0:
                scaled_pixmap = current_pixmap.scaled(
                    int(current_pixmap.width() * self.scale_factor),
                    int(current_pixmap.height() * self.scale_factor),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setPixmap(current_pixmap)
                
            self.image_label.setAlignment(Qt.AlignCenter)
            
            # Keep the zoom reset button text as "Center" regardless of scale
            self.zoom_reset_btn.setText("Center")
            
            # Update original image if in comparison mode
            if self.comparison_mode and self.original_image is not None:
                orig_height, orig_width, orig_channel = self.original_image.shape
                orig_bytes_per_line = 3 * orig_width
                orig_q_img = QImage(self.original_image.data, orig_width, orig_height, orig_bytes_per_line, QImage.Format_RGB888)
                
                # Create a pixmap from the QImage
                orig_pixmap = QPixmap.fromImage(orig_q_img)
                
                # Apply zoom if needed
                if self.scale_factor != 1.0:
                    orig_scaled_pixmap = orig_pixmap.scaled(
                        int(orig_pixmap.width() * self.scale_factor),
                        int(orig_pixmap.height() * self.scale_factor),
                        Qt.KeepAspectRatio, 
                        Qt.SmoothTransformation
                    )
                    self.original_image_label.setPixmap(orig_scaled_pixmap)
                else:
                    self.original_image_label.setPixmap(orig_pixmap)
                    
                self.original_image_label.setAlignment(Qt.AlignCenter)
            
            # Update the window title
            if self.file_path:
                self.setWindowTitle(f"Photo Editor - {os.path.basename(self.file_path)}")
                
    def save_image(self):
        if self.current_image is not None and self.file_path:
            try:
                # Start progress
                self.start_progress()
                
                # Convert RGB to BGR for OpenCV
                save_img = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
                
                # Save the image
                cv2.imwrite(self.file_path, save_img)
                
                # Update status bar
                self.status_bar.showMessage(f"Saved: {os.path.basename(self.file_path)}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not save the image: {str(e)}")
        else:
            self.save_image_as()
    
    def save_image_as(self):
        if self.current_image is not None:
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image As", "", 
                                                     "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tif *.tiff);;All Files (*)", 
                                                     options=options)
            
            if file_path:
                try:
                    # Start progress
                    self.start_progress()
                    
                    # Convert RGB to BGR for OpenCV
                    save_img = cv2.cvtColor(self.current_image, cv2.COLOR_RGB2BGR)
                    
                    # Save the image
                    cv2.imwrite(file_path, save_img)
                    
                    # Update the file path
                    self.file_path = file_path
                    
                    # Update the window title
                    self.setWindowTitle(f"Photo Editor - {os.path.basename(file_path)}")
                    
                    # Update status bar
                    self.status_bar.showMessage(f"Saved as: {os.path.basename(file_path)}")
                    
                    # End progress
                    self.end_progress()
                    
                except Exception as e:
                    self.end_progress()
                    QMessageBox.critical(self, "Error", f"Could not save the image: {str(e)}")
    
    def reset_image(self):
        if self.original_image is not None:
            # Save the current state for undo
            self.save_to_history()
            
            self.current_image = self.original_image.copy()
            self.update_image_display()
            self.status_bar.showMessage("Reset to original image")
    
    def save_to_history(self):
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            # Limit the history size
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # Enable the undo button
            self.undo_button.setEnabled(True)
    
    def undo_last_change(self):
        if self.history:
            self.current_image = self.history.pop()
            self.update_image_display()
            self.status_bar.showMessage("Undid last change")
            
            # Disable the undo button if history is empty
            if not self.history:
                self.undo_button.setEnabled(False)
        else:
            self.status_bar.showMessage("Nothing to undo")
    
    # Zoom and Pan functions
    def zoom_in(self):
        self.scale_factor *= 1.25
        self.update_image_display()
        self.status_bar.showMessage(f"Zoom: {int(self.scale_factor * 100)}%")
        
    def zoom_out(self):
        self.scale_factor /= 1.25
        if self.scale_factor < 0.1:
            self.scale_factor = 0.1
        self.update_image_display()
        self.status_bar.showMessage(f"Zoom: {int(self.scale_factor * 100)}%")
        
    def zoom_reset(self):
        self.scale_factor = 1.0
        self.update_image_display()
        self.status_bar.showMessage("Zoom: 100%")
    
    def toggle_comparison(self, enabled):
        self.comparison_mode = enabled
        if self.comparison_mode:
            self.original_image_label.show()
        else:
            self.original_image_label.hide()
        self.update_image_display()
        
    def handle_pan(self, delta):
        """Handle pan requests from the scroll area"""
        h_bar = self.scroll_area.horizontalScrollBar()
        v_bar = self.scroll_area.verticalScrollBar()
        h_bar.setValue(h_bar.value() - delta.x())
        v_bar.setValue(v_bar.value() - delta.y())
            
    def show_about(self):
        QMessageBox.about(self, "About Photo Editor", 
                          "Photo Editor\n\nA simple photo editing application based on the functions.py module.\n\nCreated with PyQt5.\n\n"
                          "Keyboard Shortcuts:\n"
                          "Ctrl+O: Open Image\n"
                          "Ctrl+S: Save Image\n"
                          "Ctrl+Z: Undo Last Change\n"
                          "Ctrl+R: Reset to Original\n"
                          "Ctrl+=: Zoom In\n"
                          "Ctrl+-: Zoom Out\n"
                          "Ctrl+0: Reset Zoom\n"
                          "Ctrl+D: Toggle Comparison\n"
                          "\nGestures:\n"
                          "Ctrl+Mouse Wheel: Zoom in/out\n"
                          "Middle-click or Ctrl+Left-click and drag: Pan\n"
                          "Touchpad pinch: Zoom in/out (if supported by system)")
    
    def start_progress(self):
        # Show progress indicator
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        # Start progress animation
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(50)
        
        # Disable UI during processing
        self.setEnabled(False)
        QApplication.processEvents()
    
    def update_progress(self):
        value = self.progress.value() + 5
        if value > 99:
            value = 0
        self.progress.setValue(value)
    
    def end_progress(self):
        # Stop timer and hide progress
        if hasattr(self, 'progress_timer'):
            self.progress_timer.stop()
        self.progress.setVisible(False)
        
        # Re-enable UI
        self.setEnabled(True)
    
    # Real-time preview functions
    def preview_brightness(self):
        if self.current_image is not None:
            # Get the value from the slider
            value = self.brightness_slider.value()
            
            # Create a preview with delayed execution to avoid too many updates
            try:
                if hasattr(self, 'preview_timer') and self.preview_timer is not None:
                    self.preview_timer.stop()
            except:
                pass
                
            self.preview_timer = QTimer()
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(lambda: self.generate_preview(lambda: brightness(self.current_image, value)))
            self.preview_timer.start(100)
    
    def preview_contrast(self):
        if self.current_image is not None:
            # Get the value from the slider
            value = self.contrast_slider.value()
            
            # Create a preview with delayed execution
            try:
                if hasattr(self, 'preview_timer') and self.preview_timer is not None:
                    self.preview_timer.stop()
            except:
                pass
                
            self.preview_timer = QTimer()
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(lambda: self.generate_preview(lambda: contrast(self.current_image, value)))
            self.preview_timer.start(100)
    
    def preview_binarization(self):
        if self.current_image is not None:
            # Get the value from the slider
            thresh = self.binary_slider.value()
            
            # Create a preview with delayed execution
            try:
                if hasattr(self, 'preview_timer') and self.preview_timer is not None:
                    self.preview_timer.stop()
            except:
                pass
                
            self.preview_timer = QTimer()
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(lambda: self.generate_preview(lambda: self.convert_to_rgb(binarize(self.current_image, thresh))))
            self.preview_timer.start(100)
    
    def preview_sharpness(self):
        if self.current_image is not None:
            # Get the value from the slider
            strength = self.sharp_slider.value()
            
            # Create a preview with delayed execution
            try:
                if hasattr(self, 'preview_timer') and self.preview_timer is not None:
                    self.preview_timer.stop()
            except:
                pass
                
            self.preview_timer = QTimer()
            self.preview_timer.setSingleShot(True)
            self.preview_timer.timeout.connect(lambda: self.generate_preview(lambda: sharpness(self.current_image, strength)))
            self.preview_timer.start(150)
    
    def generate_preview(self, processing_func):
        try:
            # Process the image
            preview = processing_func()
            
            # Update display with preview
            self.update_preview_display(preview)
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview error: {str(e)}")
    
    def update_preview_display(self, preview):
        if preview is not None:
            # Convert the numpy array to QImage
            height, width = preview.shape[:2]
            
            if len(preview.shape) == 2:  # Grayscale
                q_img = QImage(preview.data, width, height, width, QImage.Format_Grayscale8)
            else:  # RGB
                bytes_per_line = 3 * width
                q_img = QImage(preview.data, width, height, bytes_per_line, QImage.Format_RGB888)
            
            # Create a pixmap from the QImage
            pixmap = QPixmap.fromImage(q_img)
            
            # Apply zoom if needed
            if self.scale_factor != 1.0:
                scaled_pixmap = pixmap.scaled(
                    int(pixmap.width() * self.scale_factor),
                    int(pixmap.height() * self.scale_factor),
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                self.image_label.setPixmap(scaled_pixmap)
            else:
                self.image_label.setPixmap(pixmap)
                
            self.image_label.setAlignment(Qt.AlignCenter)
    
    def convert_to_rgb(self, img):
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img
    
    # Image processing functions
    def apply_grayscale(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                method = self.grayscale_combo.currentText()
                
                if method == "Luminance":
                    result = grayscale_luminance(self.current_image)
                elif method == "Lightness":
                    result = grayscale_lightness(self.current_image)
                elif method == "Average":
                    result = grayscale_average(self.current_image)
                
                # Convert to 3-channel image for display
                self.current_image = self.convert_to_rgb(result)
                
                self.update_image_display()
                self.status_bar.showMessage(f"Applied grayscale ({method})")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply grayscale: {str(e)}")
    
    def apply_brightness(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                value = self.brightness_slider.value()
                self.current_image = brightness(self.current_image, value)
                self.update_image_display()
                self.status_bar.showMessage(f"Applied brightness adjustment: {value}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply brightness: {str(e)}")
    
    def apply_contrast(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                value = self.contrast_slider.value()
                self.current_image = contrast(self.current_image, value)
                self.update_image_display()
                self.status_bar.showMessage(f"Applied contrast adjustment: {value}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply contrast: {str(e)}")
    
    def apply_inversion(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                self.current_image = inverse(self.current_image)
                self.update_image_display()
                self.status_bar.showMessage("Applied color inversion")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply inversion: {str(e)}")
    
    def apply_binarization(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                thresh = self.binary_slider.value()
                result = binarize(self.current_image, thresh)
                
                # Convert to 3-channel image for display
                self.current_image = self.convert_to_rgb(result)
                
                self.update_image_display()
                self.status_bar.showMessage(f"Applied binarization with threshold: {thresh}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply binarization: {str(e)}")
    
    def apply_sharpness(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                strength = self.sharp_slider.value()
                self.current_image = sharpness(self.current_image, strength)
                self.update_image_display()
                self.status_bar.showMessage(f"Applied sharpness with strength: {strength}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply sharpness: {str(e)}")
    
    def apply_mean_filter(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                k = self.mean_kernel.value()
                # Ensure k is odd
                if k % 2 == 0:
                    k += 1
                    self.mean_kernel.setValue(k)
                
                self.current_image = mean_filter(self.current_image, k)
                self.update_image_display()
                self.status_bar.showMessage(f"Applied mean filter with kernel size: {k}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply mean filter: {str(e)}")
    
    def apply_gaussian_blur(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                k = self.gauss_kernel.value()
                sigma = self.gauss_sigma.value()
                
                self.current_image = gaussian_blur(self.current_image, k, sigma)
                self.update_image_display()
                self.status_bar.showMessage(f"Applied Gaussian blur with kernel size: {k}, sigma: {sigma}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply Gaussian blur: {str(e)}")
    
    # NEW: Apply median filter
    def apply_median_filter(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                # Get kernel size
                k = self.median_kernel.value()
                # Ensure k is odd
                if k % 2 == 0:
                    k += 1
                    self.median_kernel.setValue(k)
                
                # Apply median filter
                self.current_image = median_filter(self.current_image, k)
                self.update_image_display()
                self.status_bar.showMessage(f"Applied median filter with kernel size: {k}")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply median filter: {str(e)}")

    # NEW: Apply edge detection
    def apply_edge_detection(self):
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
                
                # Start progress
                self.start_progress()
                
                # Get selected method
                method = self.edge_method_combo.currentText()
                
                # Apply the selected edge detection method
                if method == "Roberts Cross":
                    result = roberts_cross(self.current_image)
                
                elif method == "Prewitt":
                    # Get selected direction (extract numeric value from string)
                    direction_str = self.prewitt_direction_combo.currentText()
                    direction = int(direction_str.split(':')[0])
                    result = prewitt(self.current_image, direction)
                
                elif method == "Prewitt (All Directions)":
                    magnitude, direction_map = prewitt_all_directions(self.current_image)
                    # Display only the magnitude for now
                    result = magnitude
                    # Could also show direction map in a separate window if desired
                
                elif method == "Prewitt (Gradient Magnitude)":
                    result = prewitt_gradient_magnitude(self.current_image)
                
                elif method == "Sobel":
                    # Get selected direction (extract numeric value from string)
                    direction_str = self.sobel_direction_combo.currentText()
                    direction = int(direction_str.replace('°', ''))
                    result = sobel(self.current_image, direction)
                
                elif method == "Sobel (Gradient Magnitude)":
                    magnitude, direction_map = sobel_gradient_magnitude(self.current_image)
                    # Display only the magnitude for now
                    result = magnitude
                    # Could also visualize direction map if desired
                
                # Convert to 3-channel image for display if needed
                if len(result.shape) == 2:
                    self.current_image = self.convert_to_rgb(result)
                else:
                    self.current_image = result
                
                self.update_image_display()
                self.status_bar.showMessage(f"Applied {method} edge detection")
                
                # End progress
                self.end_progress()
                
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply edge detection: {str(e)}")
    
    def show_histogram(self):
        if self.current_image is not None:
            try:
                # Use the plot_histogram function from functions.py
                plot_histogram(self.current_image)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not show histogram: {str(e)}")
    
    def show_projections(self):
        if self.current_image is not None:
            try:
                # Convert to grayscale if not already
                if len(self.current_image.shape) == 3:
                    gray_image = grayscale_luminance(self.current_image)
                else:
                    gray_image = self.current_image
                
                # Use the plot_image_with_projections function from functions.py
                plot_image_with_projections(gray_image)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not show projections: {str(e)}")
                
    # Custom kernel methods
    def update_kernel_grid(self):
        """Update the kernel grid based on the selected size"""
        # Clear existing grid
        for i in reversed(range(self.kernel_grid_layout.count())):
            widget = self.kernel_grid_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
    
        # Get selected size
        size_text = self.kernel_size_combo.currentText()
        size = int(size_text.split('x')[0])
    
        # Create grid of input fields
        self.kernel_inputs = []
        for i in range(size):
            row_inputs = []
            for j in range(size):
                spin_box = QDoubleSpinBox()
                spin_box.setRange(-10.0, 10.0)  # Reasonable range for kernel values
                spin_box.setSingleStep(0.1)
                spin_box.setValue(0.0)  # Default value
                if i == size//2 and j == size//2:  # Center value
                    spin_box.setValue(1.0)  # Identity kernel by default
            
                self.kernel_grid_layout.addWidget(spin_box, i, j)
                row_inputs.append(spin_box)
            self.kernel_inputs.append(row_inputs)

    def get_kernel_values(self):
        """Get the kernel values from the input grid"""
        size = len(self.kernel_inputs)
        kernel = np.zeros((size, size), dtype=float)
    
        for i in range(size):
            for j in range(size):
                kernel[i, j] = self.kernel_inputs[i][j].value()
    
        # Normalize if checked
        if self.normalize_kernel_check.isChecked():
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:  # Avoid division by zero
                kernel = kernel / kernel_sum
    
        return kernel

    def apply_custom_kernel(self):
        """Apply the custom kernel to the image"""
        if self.current_image is not None:
            try:
                # Save the current state for undo
                self.save_to_history()
            
                # Start progress
                self.start_progress()
            
                # Get kernel values
                kernel = self.get_kernel_values()
            
                # Apply the kernel
                self.current_image = apply_custom_kernel(self.current_image, kernel)
            
                # Update display
                self.update_image_display()
                self.status_bar.showMessage(f"Applied custom {kernel.shape[0]}x{kernel.shape[1]} kernel")
            
                # End progress
                self.end_progress()
            
            except Exception as e:
                self.end_progress()
                QMessageBox.critical(self, "Error", f"Could not apply custom kernel: {str(e)}")

    def apply_predefined_kernel(self):
        """Apply a predefined kernel template"""
        current_preset = self.predefined_combo.currentText()
    
        # Reset to custom if selecting Custom
        if current_preset == "Custom":
            return
    
        # Get current kernel size
        size_text = self.kernel_size_combo.currentText()
        size = int(size_text.split('x')[0])
    
        # Define preset kernels for different sizes
        if current_preset == "Blur":
            # Blur kernel (averaging)
            for i in range(size):
                for j in range(size):
                    self.kernel_inputs[i][j].setValue(1.0 / (size * size))
                
        elif current_preset == "Sharpen":
            # Reset all to 0
            for i in range(size):
                for j in range(size):
                    self.kernel_inputs[i][j].setValue(0.0)
        
            # For any size, create a sharpen kernel
            center = size // 2
            # Set center value
            self.kernel_inputs[center][center].setValue(2.0)
        
            # Set adjacent values (cross pattern)
            if center > 0:
                self.kernel_inputs[center-1][center].setValue(-0.25)
            if center < size-1:
                self.kernel_inputs[center+1][center].setValue(-0.25)
            if center > 0:
                self.kernel_inputs[center][center-1].setValue(-0.25)
            if center < size-1:
                self.kernel_inputs[center][center+1].setValue(-0.25)
            
        elif current_preset == "Edge Detection":
            # Sobel-like edge detection
            # Reset all to 0
            for i in range(size):
                for j in range(size):
                    self.kernel_inputs[i][j].setValue(0.0)
                
            if size == 3:
                # 3x3 Sobel-like
                self.kernel_inputs[0][0].setValue(-1.0)
                self.kernel_inputs[0][1].setValue(-1.0)
                self.kernel_inputs[0][2].setValue(-1.0)
                self.kernel_inputs[1][0].setValue(-1.0)
                self.kernel_inputs[1][1].setValue(8.0)
                self.kernel_inputs[1][2].setValue(-1.0)
                self.kernel_inputs[2][0].setValue(-1.0)
                self.kernel_inputs[2][1].setValue(-1.0)
                self.kernel_inputs[2][2].setValue(-1.0)
            else:
                # For larger kernels, set a basic edge detection pattern
                center = size // 2
                for i in range(size):
                    for j in range(size):
                        if i == center and j == center:
                            self.kernel_inputs[i][j].setValue(size * size - 1)
                        else:
                            self.kernel_inputs[i][j].setValue(-1.0)
                        
        elif current_preset == "Emboss":
            # Emboss effect
            # Reset all to 0
            for i in range(size):
                for j in range(size):
                    self.kernel_inputs[i][j].setValue(0.0)
                
            if size == 3:
                # 3x3 emboss
                self.kernel_inputs[0][0].setValue(-2.0)
                self.kernel_inputs[0][1].setValue(-1.0)
                self.kernel_inputs[0][2].setValue(0.0)
                self.kernel_inputs[1][0].setValue(-1.0)
                self.kernel_inputs[1][1].setValue(1.0)
                self.kernel_inputs[1][2].setValue(1.0)
                self.kernel_inputs[2][0].setValue(0.0)
                self.kernel_inputs[2][1].setValue(1.0)
                self.kernel_inputs[2][2].setValue(2.0)
            else:
                # For larger kernels, create a diagonal gradient
                for i in range(size):
                    for j in range(size):
                        val = (j - i) * (2.0 / (size - 1))
                        self.kernel_inputs[i][j].setValue(val)
    
        # Uncheck normalization for edge detection and emboss
        if current_preset in ["Edge Detection", "Emboss"]:
            self.normalize_kernel_check.setChecked(False)
        else:
            self.normalize_kernel_check.setChecked(True)
    
        # Reset to custom to avoid reapplying
        self.predefined_combo.setCurrentIndex(0)


def apply_style(app, font_size=9):
    # Save current font family before changing styles
    current_font = app.font()
    font_family = current_font.family()
    
    # Set a dark theme
    app.setStyle("Fusion")
    
    # Create a font with the specified size
    font = QFont(font_family)
    font.setPointSize(font_size)
    app.setFont(font)
    
    # Set a dark color palette
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.WindowText, Qt.white)
    dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
    dark_palette.setColor(QPalette.ToolTipText, Qt.white)
    dark_palette.setColor(QPalette.Text, Qt.white)
    dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ButtonText, Qt.white)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.HighlightedText, Qt.black)
    
    app.setPalette(dark_palette)
    
    # Set stylesheet for more control
    app.setStyleSheet(f"""
        QMainWindow {{
            background-color: #303030;
        }}
        QGroupBox {{
            font-weight: bold;
            border: 1px solid #606060;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
        }}
        QPushButton {{
            background-color: #4d6c8b;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            min-height: {font_size * 2}px;
        }}
        QPushButton:hover {{
            background-color: #607c9b;
        }}
        QPushButton:pressed {{
            background-color: #3d5c7b;
        }}
        QSlider::groove:horizontal {{
            border: 1px solid #999999;
            height: 8px;
            background: #4d4d4d;
            margin: 2px 0;
            border-radius: 4px;
        }}
        QSlider::handle:horizontal {{
            background: #4d6c8b;
            border: 1px solid #5c7b9b;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }}
        QTabWidget::pane {{
            border: 1px solid #444;
            border-radius: 3px;
            top: -1px;
            background-color: #303030;
        }}
        QTabBar::tab {{
            background-color: #404040;
            color: white;
            padding: 8px 15px;
            border-top-left-radius: 3px;
            border-top-right-radius: 3px;
        }}
        QTabBar::tab:selected {{
            background-color: #4d6c8b;
        }}
        QTabBar::tab:hover {{
            background-color: #505050;
        }}
        QSpinBox, QDoubleSpinBox {{
            background-color: #404040;
            color: white;
            border: 1px solid #505050;
            border-radius: 3px;
            padding: 2px 5px;
            min-height: {font_size * 1.8}px;
        }}
        QComboBox {{
            background-color: #404040;
            color: white;
            border: 1px solid #505050;
            border-radius: 3px;
            padding: 2px 5px;
            min-height: {font_size * 1.8}px;
        }}
        QLabel {{
            color: white;
        }}
        QMenuBar {{
            background-color: #404040;
            color: white;
        }}
        QMenuBar::item:selected {{
            background-color: #4d6c8b;
        }}
        QMenu {{
            background-color: #404040;
            color: white;
        }}
        QMenu::item:selected {{
            background-color: #4d6c8b;
        }}
        QScrollArea {{
            background-color: #303030;
            border: 1px solid #404040;
            border-radius: 3px;
        }}
        QStatusBar {{
            background-color: #404040;
            color: white;
        }}
        QToolBar {{
            background-color: #404040;
            border: none;
            spacing: 3px;
            padding: 3px;
        }}
        QToolBar QToolButton {{
            background-color: transparent;
            border-radius: 3px;
            padding: 3px;
        }}
        QToolBar QToolButton:hover {{
            background-color: #505050;
        }}
        QToolBar QToolButton:pressed {{
            background-color: #3d5c7b;
        }}
        QCheckBox {{
            color: white;
            spacing: 5px;
        }}
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border-radius: 2px;
        }}
        QCheckBox::indicator:unchecked {{
            background-color: #404040;
            border: 1px solid #606060;
        }}
        QCheckBox::indicator:checked {{
            background-color: #4d6c8b;
            border: 1px solid #4d6c8b;
        }}
        QFontDialog {{
            background-color: #404040;
        }}
    """)

def main():
    # Enable high DPI scaling before creating QApplication
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Load user settings if available
    settings = QSettings("PhotoEditor", "FontSettings")
    font_size = settings.value("font_size", 9, type=int)
    
    # Enable general touch input support
    app.setAttribute(Qt.AA_SynthesizeMouseForUnhandledTouchEvents, True)
    app.setAttribute(Qt.AA_CompressHighFrequencyEvents, True)
    
    # Apply the dark theme with custom font size
    apply_style(app, font_size)
    
    window = PhotoEditor()
    
    # Get available screen geometry
    screen = app.primaryScreen()
    screen_geometry = screen.availableGeometry()
    screen_width = screen_geometry.width()
    screen_height = screen_geometry.height()
    
    # Center the window on screen with appropriate size
    window.setGeometry(
        (screen_width - int(screen_width * 0.8)) // 2,
        (screen_height - int(screen_height * 0.8)) // 2,
        int(screen_width * 0.8),
        int(screen_height * 0.8)
    )
    
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()