import sys
import time
import cv2
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor
from PyQt5.QtWidgets import (
    QTableWidget, QMessageBox, QDialog, QListWidget, QPushButton, QMenu, QFileDialog,
    QOpenGLWidget, QSizePolicy, QTableWidgetItem, QHeaderView, QApplication, QMainWindow,QScrollArea,
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QScrollBar
)
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import math
from scipy.spatial.transform import Rotation
import os
import datetime
import time
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def quaternion_to_euler(x, y, z, w):
    rot = Rotation.from_quat([x, y, z, w])
    rot_euler = rot.as_euler('zyx', degrees=True)
    return rot_euler[2], rot_euler[1], rot_euler[0]


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("""
            QMainWindow {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 navy, stop:1 cyan);
                font-family: Armstrong;
            }
            QLabel, QPushButton, QTableWidget, QHeaderView, QListWidget {
                font-family: Armstrong;
                font-size: 14px;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """)

        self.central_widget = QLabel(self)
        self.setCentralWidget(self.central_widget)

        # Değişkenler eklendi
        self.player_video_loaded = False
        self.ball_drop_point_video_loaded = False

        self.player_video_cap = None
        self.ball_drop_point_video_cap = None

         # Initialize the start time when the application starts
        self.start_time = None

        self.success_label = QLabel(self)
        #self.success_label.setFixedSize(230, 540)
        self.success_label.move(1000, 10)
        self.success_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "  
            "border-radius: 8px; padding: 5px; "  
            "font-size: 14px; color: #003366;"  
        )
        self.success_label.setText(
            "Success Rate\n\n"
        )
        self.success_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.success_label.setWordWrap(True)  # Metnin satır kaydırması için

        # Sabit boyutu kaldır, adjustSize ile büyümesine izin ver
        self.success_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Scroll Area oluşturuluyor ve QLabel içine yerleştirilme
        self.success_scroll_area = QScrollArea(self)
        self.success_scroll_area.setFixedSize(230, 540)
        self.success_scroll_area.move(1000, 10)
        self.success_scroll_area.setWidget(self.success_label)
        self.success_scroll_area.setWidgetResizable(True)
        self.success_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.success_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.success_label.adjustSize()


        # QScrollArea'nın stilini QLabel ile uyumlu hale getirme
        self.success_scroll_area.setStyleSheet(
            "QScrollArea {"
            "background-color: rgba(255, 255, 255, 200);"  
            "border-radius: 8px; "  
            "padding: 5px;"  
            "border: none;"  
            "}"
            "QScrollBar:vertical {"
            "    width: 8px;"
            "    margin: 2px 0 2px 0;"
            "    border-radius: 4px;"
            "}"
        )


        # QLabel for graph
        self.graph_label = QLabel(self)
        self.graph_label.setFixedSize(500, 415)
        self.graph_label.move(1280, 450)
        self.graph_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "
            "border-radius: 8px; padding: 5px; "
            "font-size: 14px; color: #003366;"
        )

        # QLabel for heatmap
        self.heatmap_label = QLabel(self)
        self.heatmap_label.setFixedSize(650, 300)
        self.heatmap_label.move(580, 560)
        self.heatmap_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "
            "border-radius: 8px; padding: 5px; "
            "font-size: 14px; color: #003366;"
        )

        # Matplotlib figure for heatmap
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.heatmap_ax = self.figure.add_subplot(111)

        heatmap_layout = QVBoxLayout(self.heatmap_label)
        heatmap_layout.addWidget(self.canvas)

        self.plot_heatmap()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.graph_ax = self.figure.add_subplot(111)
        self.graph_ax.set_xlabel('Time (s)')
        self.graph_ax.set_ylabel('Angle (degrees)')
        self.graph_ax.legend(['Pitch', 'Yaw', 'Roll'])
        self.graph_ax.grid(True)

        self.angle_data = {'pitch': [], 'yaw': [], 'roll': [], 'time': []}

        # Create a layout for the graph and scrollbar
        graph_layout = QVBoxLayout(self.graph_label)
        graph_layout.addWidget(self.canvas)

        # Add a horizontal scrollbar for the graph
        self.scrollbar = QScrollBar(Qt.Horizontal, self)
        graph_layout.addWidget(self.scrollbar)

        self.scrollbar.setRange(0, 100)
        self.scrollbar.setPageStep(10)
        self.scrollbar.valueChanged.connect(self.update_graph_view)

        # Unified label for ball drop point information
        self.timestamp_conversion_label = QLabel(self)
        self.timestamp_conversion_label.setFixedSize(500, 60) 
        self.timestamp_conversion_label.move(580, 900)
        self.timestamp_conversion_label.setText(
            "Ball Drop Point\nx_line: \ny_line: "
        )
        self.timestamp_conversion_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "
            "border-radius: 8px; padding: 5px; "
            "font-size: 14px; color: #003366;"
        )

        self.player_conversion_info = "Timestamp-Time Conversion for Player: "
        self.ball_conversion_info = "Timestamp-Time Conversion for Ball: "

        # Unified label for ball drop point information
        self.unified_frame_info_label = QLabel(self)
        self.unified_frame_info_label.setFixedSize(500, 60)
        self.unified_frame_info_label.move(1280, 900)
        self.unified_frame_info_label.setText(
            "Frame-Timestamp Information\nFrame:     | Timestamp:       "
        )
        self.unified_frame_info_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "
            "border-radius: 8px; padding: 5px; "
            "font-size: 14px; color: #003366;"
        )

        self.ball_drop_info = "Frame-Timestamp Information - Frame: | Timestamp: "

        self.player_timestamp_file = ""
        self.ball_drop_point_timestamp_file = ""
        self.player_frame_timestamps = {}
        self.ball_drop_point_frame_timestamps = {}

        # Label for training time (updated with player video only)
        self.total_training_time_label = QLabel(self)
        self.total_training_time_label.setFixedSize(400, 60)
        self.total_training_time_label.move(10, 900)
        self.total_training_time_label.setText(
            "Total Training Time: 00:00:00\nDate: YYYY-MM-DD"
        )
        self.total_training_time_label.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "
            "border-radius: 8px; padding: 5px; "
            "font-size: 14px; color: #003366;"
        )

        self.player_label = QLabel(self)
        self.player_label.setFixedSize(720, 540)
        self.player_label.move(10, -80)

        self.ball_drop_point_label = QLabel(self)
        self.ball_drop_point_label.setFixedSize(720, 540)
        self.ball_drop_point_label.move(10, 560)

        self.selected_svo_player = ""
        self.selected_svo_ball_drop_point = ""

        self.svo_video_player = sl.Camera()
        self.svo_video_ball_drop_point = sl.Camera()

        self.play_both_button = QPushButton('Play All', self)
        self.play_both_button.move(1800, 60)
        self.play_both_button.clicked.connect(self.play_both_videos)

        self.play_pause_button = QPushButton('Play/Pause', self)
        self.play_pause_button.move(1800, 100)
        self.play_pause_button.clicked.connect(self.toggle_play_pause)

        self.is_playing = False

        self.active_view = None
        self.selected_cameras = []
        self.disable_body_tracking = False

        self.question_mark_button = QPushButton('?', self)
        self.question_mark_button.setFont(QFont('Arial', 14))
        self.question_mark_button.setFixedSize(30, 30)
        self.question_mark_button.move(1835, 140)
        self.question_mark_button.clicked.connect(self.show_image_popup)

        self.setWindowTitle(
            "TÜBİTAK 1001 - Teniste Yapay Zeka Destekli Sanal Spor Koçu"
        )

        self.image_label = QLabel(self)
        self.image_label.setFixedSize(720, 540)
        self.image_label.move(1280, 0)

        self.image_label_2 = QLabel(self)
        self.image_label_2.setFixedSize(720, 540)
        self.image_label_2.move(570, -103)

        self.table_widget = QTableWidget(self)
        self.table_widget.setColumnCount(4)
        self.table_widget.setHorizontalHeaderLabels(['Index', 'Pitch', 'Yaw', 'Roll'])

        # Sütun genişliğini ayarlayın
        self.table_widget.setColumnWidth(0, 150)  # Index sütunu için genişlik

        self.table_widget.setFixedSize(500, 415)
        self.table_widget.move(1280, 10)

        # Diğer sütunlar için otomatik genişleme
        self.table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Index sütun genişliği sabit
        self.table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Pitch sütunu
        self.table_widget.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Yaw sütunu
        self.table_widget.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)  # Roll sütunu

        self.table_widget.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Dikey başlığı gizle
        self.table_widget.verticalHeader().setVisible(False)


        # Set style for the table
        self.table_widget.setStyleSheet(
            "background-color: rgba(255, 255, 255, 200); "
            "border-radius: 8px; padding: 5px; "
            "font-size: 14px; color: #003366;"
        )

        self.table_widget.cellClicked.connect(self.on_table_cell_clicked)
        self.selected_row = -1  # Store the currently selected row

        self.start_time = None  # To track when the row selection or video play started

        self.graph_dialogs = {}

        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True
        self.body_param.enable_body_fitting = True
        self.body_param.detection_model = (
            sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        )
        self.body_param.body_format = sl.BODY_FORMAT.BODY_34

        self.svo_video = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params.svo_real_time_mode = True
        self.svo_video.open(self.init_params)
        self.svo_video.enable_positional_tracking(
            sl.PositionalTrackingParameters()
        )
        self.svo_video.enable_body_tracking(self.body_param)

        self.zed = sl.Camera()
        self.init_params_2 = sl.InitParameters()
        self.init_params_2.camera_resolution = sl.RESOLUTION.HD720
        self.init_params_2.coordinate_units = sl.UNIT.METER
        self.init_params_2.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params_2.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        self.init_params_2.svo_real_time_mode = True
        self.zed.open(self.init_params_2)
        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        self.zed.enable_body_tracking(self.body_param)

        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 40

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_view)

        self.central_widget.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.central_widget.setFixedSize(1280, 720)

        self.play_button = QPushButton('Choose', self)
        self.play_button.move(1800, 20)
        self.play_button.clicked.connect(self.show_play_options)
        self.play_menu = QMenu(self)
        self.action_play_svo = self.play_menu.addAction('Record', self.play_svo)
        self.action_play_zed = self.play_menu.addAction('Live', self.play_zed_live)

        image_path = "C:\\Users\\Buse\\Desktop\\angles.jpeg"
        image_pixmap = QPixmap(image_path)
        new_height = 480
        aspect_ratio = image_pixmap.width() / image_pixmap.height()
        new_width = int(new_height * aspect_ratio)
        scaled_pixmap = image_pixmap.scaled(new_width, new_height, Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.setFixedSize(new_width, new_height)
        self.image_label.move(1280, 450)
        self.image_label.setVisible(False)

        # Add logo labels
        self.logo_label_1 = QLabel(self)
        self.logo_label_1.setFixedSize(100, 100)
        self.logo_label_1.move(1800, 890)
        logo_1_pixmap = QPixmap(r"C:\Users\Buse\Desktop\Asil Logo")
        self.logo_label_1.setPixmap(
            logo_1_pixmap.scaled(self.logo_label_1.size(), Qt.KeepAspectRatio)
        )

        self.logo_label_2 = QLabel(self)
        self.logo_label_2.setFixedSize(100, 100)
        self.logo_label_2.move(1810, 790)
        logo_1_pixmap = QPixmap(r"C:\Users\Buse\Desktop\tübitak")
        self.logo_label_2.setPixmap(
            logo_1_pixmap.scaled(self.logo_label_2.size(), Qt.KeepAspectRatio)
        )

        self.logo_label_3 = QLabel(self)
        self.logo_label_3.setFixedSize(100, 100)
        self.logo_label_3.move(1805, 690)
        logo_1_pixmap = QPixmap(r"C:\Users\Buse\Desktop\mta")
        self.logo_label_3.setPixmap(
            logo_1_pixmap.scaled(self.logo_label_3.size(), Qt.KeepAspectRatio)
        )

        # Initialize heatmap storage
        #self.heatmap = np.zeros((540, 720), dtype=np.float32)

    def toggle_play_pause(self):
        if self.is_playing:
            self.timer.stop()
            self.play_pause_button.setText('Play')
        else:
            self.timer.start(30)
            self.play_pause_button.setText('Pause')
        self.is_playing = not self.is_playing


    def update_success_label(self, new_data):
        current_text = self.success_label.text()
        updated_text = current_text + f"{new_data}\n\n"  # Extra newline for spacing between frames
        self.success_label.setText(updated_text)

         # İçeriğin büyüklüğüne göre QLabel'in boyutunu ayarla
        self.success_label.adjustSize()

        # Scrollbar'ı en aşağıya kaydır
        QTimer.singleShot(0, lambda: self.success_scroll_area.verticalScrollBar().setValue(
            self.success_scroll_area.verticalScrollBar().maximum()))


    def on_table_cell_clicked(self, row, column):
        self.selected_row = row  # Update the selected row index
        if self.start_time is None:
            # Set start_time only if it's not already set
            self.start_time = time.time()  # Capture the start time when a row is selected
        self.update_graph_with_selected_row(row)

    def load_vurus_frames(filepath):
        vurus_frames = []
        with open(filepath, 'r') as file:
            for line in file:
                vurus_frames.append(int(line.strip()))  # Her satırdaki frame numarasını listeye ekle
        return vurus_frames

    def update_graph_with_selected_row(self, row):
        if row == -1:
            return
        
        # Seçilen satırdan verileri al
        pitch_data = float(self.table_widget.item(row, 1).text())
        yaw_data = float(self.table_widget.item(row, 2).text())
        roll_data = float(self.table_widget.item(row, 3).text())

         # Ensure start_time is initialized
        if self.start_time is None:
            self.start_time = time.time()

        # Zamanı güncelle
        current_time = time.time() - self.start_time
        self.angle_data['time'].append(current_time)
        self.angle_data['pitch'].append(pitch_data)
        self.angle_data['yaw'].append(yaw_data)
        self.angle_data['roll'].append(roll_data)
        
        # Liste uzunluklarının aynı olduğundan emin ol
        min_length = min(len(self.angle_data['time']), len(self.angle_data['pitch']), len(self.angle_data['yaw']), len(self.angle_data['roll']))
        self.angle_data['time'] = self.angle_data['time'][:min_length]
        self.angle_data['pitch'] = self.angle_data['pitch'][:min_length]
        self.angle_data['yaw'] = self.angle_data['yaw'][:min_length]
        self.angle_data['roll'] = self.angle_data['roll'][:min_length]

        self.plot_graph()  # Grafiği güncelle

    def plot_graph(self):
        #Update the graph with the selected row's data and show the index
        self.graph_ax.clear()
        self.graph_ax.set_xlabel('Time (s)')
        self.graph_ax.set_ylabel('Angle (degrees)')
        self.graph_ax.grid(True)

        if len(self.angle_data['time']) > 0 and len(self.angle_data['pitch']) > 0:
            min_length = min(len(self.angle_data['time']), len(self.angle_data['pitch']), len(self.angle_data['yaw']), len(self.angle_data['roll']))
            time_data = self.angle_data['time'][:min_length]
            pitch_data = self.angle_data['pitch'][:min_length]
            yaw_data = self.angle_data['yaw'][:min_length]
            roll_data = self.angle_data['roll'][:min_length]

            self.graph_ax.plot(time_data, pitch_data, label='Pitch', color='red')
            self.graph_ax.plot(time_data, yaw_data, label='Yaw', color='green')
            self.graph_ax.plot(time_data, roll_data, label='Roll', color='blue')

        # Update the title to include the selected index
        if self.selected_row != -1:
            selected_index = self.table_widget.item(self.selected_row, 0).text()
            self.graph_ax.set_title(f"Selected index is: {selected_index}")

        self.graph_ax.legend()

        if len(self.angle_data['time']) > 0:
            max_time = max(self.angle_data['time'])
            view_window_size = 10
            self.scrollbar.setMaximum(int(max_time))
            self.scrollbar.setValue(int(max_time) - view_window_size)
            self.graph_ax.set_xlim(max(0, max_time - view_window_size), max_time)

        self.canvas.draw()

    def update_graph_view(self):
        """Update the graph view based on scrollbar value."""
        # Get the current scroll value
        scroll_value = self.scrollbar.value()

        # Calculate the view window size
        view_window_size = 10
        max_time_range = max(self.angle_data['time'], default=0)

        # Dynamically set the range of the scrollbar
        if max_time_range > 0:
            self.scrollbar.setRange(0, int(max_time_range) - view_window_size)
            self.scrollbar.setPageStep(view_window_size)

        # Set the x-axis limits based on scrollbar position
        self.graph_ax.set_xlim(scroll_value, scroll_value + view_window_size)
        self.canvas.draw()

    def show_graph_popup(self, row, column):
        index_value = self.table_widget.item(row, 0).text()
        pitch = float(self.table_widget.item(row, 1).text())
        yaw = float(self.table_widget.item(row, 2).text())
        roll = float(self.table_widget.item(row, 3).text())

        self.graph_ax.clear()
        self.graph_ax.set_xlabel('Açı Türü')
        self.graph_ax.set_ylabel('Açı (derece)')
        self.graph_ax.grid(True)
        self.graph_ax.bar(['Pitch', 'Yaw', 'Roll'], [pitch, yaw, roll], color=['red', 'green', 'blue'])
        self.canvas.draw()

    def update_total_training_time(self, elapsed_time):
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.total_training_time_label.setText(
            f"Total Training Time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}\nDate: {current_date}"
        )

    def show_image_popup(self):
        msg_box = QDialog(self)
        msg_box.setWindowTitle("Angles Image")
        msg_box.setFixedSize(500, 500)
        layout = QVBoxLayout()
        label = QLabel(msg_box)

        pixmap = QPixmap('C:/Users/Buse/Desktop/angles.jpeg')
        label.setPixmap(pixmap)
        label.setScaledContents(True)
        layout.addWidget(label)

        msg_box.setLayout(layout)
        msg_box.exec_()

    def load_timestamps(self, filepath, video_type):
        timestamps = {}
        with open(filepath, 'r') as file:
            next(file)  # Skip header if present
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    frame = int(parts[0])
                    timestamp = float(parts[1]) / 1e9  # Nanoseconds to seconds
                    timestamps[frame] = timestamp  # Store frame-timestamp pairs

        if video_type == "player":
            self.player_frame_timestamps = timestamps
        elif video_type == "ball_drop_point":
            self.ball_drop_point_frame_timestamps = timestamps


    def show_camera_selection_dialog(self):
        camera_list = sl.Camera.get_device_list()
        dialog = CameraSelectionDialog(camera_list, self)
        if dialog.exec_() == QDialog.Accepted:
            self.selected_cameras = dialog.selected_cameras

    def show_play_options(self):
        self.play_menu.exec_(self.play_button.mapToGlobal(QPoint(0, 0)))

    def show_svo_options(self):
        svo_menu = QMenu("SVO Seçenekleri", self)
        self.action_play_player = svo_menu.addAction('Oyuncu', self.play_svo_player)
        self.action_play_ball_drop_point = svo_menu.addAction(
            'Top Düşme Noktası', self.play_svo_ball_drop_point
        )
        svo_menu.exec_(self.play_button.mapToGlobal(QPoint(0, 0)))

    def set_video_type(self, video_type):
        self.current_video_type = video_type

    # Add this method to check if a file is an SVO file
    def is_svo_file(self, file_path):
        return file_path.endswith('.svo')

    def play_svo_player(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Player Video",
            "",
            "SVO files (*.svo);;MP4 files (*.mp4);;AVI files (*.avi)",
        )
        if video_file:
            self.selected_svo_player = video_file
            if video_file.endswith('.svo'):
                self.player_video_loaded = True
            elif video_file.endswith(('.mp4', '.avi')):
                txt_file, _ = QFileDialog.getOpenFileName(
                    self, "Select Timestamp File", "", "TXT files (*.txt)"
                )
                if txt_file:
                    self.player_timestamp_file = txt_file
                    self.load_timestamps(txt_file, "player")
                    self.player_video_loaded = True

            # Get the creation date of the video file
            creation_time = os.path.getctime(video_file)
            self.creation_date_player = datetime.datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d")
            
            # Update the total training time label with the video creation date
            self.total_training_time_label.setText(
                f"Total Training Time: 00:00:00\nDate: {self.creation_date_player}"
            )


    def play_svo_ball_drop_point(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ball Drop Point Video",
            "",
            "SVO files (*.svo);;MP4 files (*.mp4);;AVI files (*.avi)",
        )
        if video_file:
            self.selected_svo_ball_drop_point = video_file
            if video_file.endswith('.avi'):
                txt_file, _ = QFileDialog.getOpenFileName(
                    self, "Select Timestamp File", "", "TXT files (*.txt)"
                )
                if txt_file:
                    self.ball_drop_point_timestamp_file = txt_file
                    self.load_timestamps(txt_file, "ball_drop_point")
                    self.ball_drop_point_video_loaded = True


    def play_svo_ball_drop_point(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self,
            "Topun düşme noktasını gösteren videoyu seçin",
            "",
            "SVO files (*.svo);;MP4 files (*.mp4);;AVI files (*.avi)",
        )
        if video_file:
            self.selected_svo_ball_drop_point = video_file
            if video_file.endswith('.avi'):
                txt_file, _ = QFileDialog.getOpenFileName(
                    self, "Timestamp dosyasını seçin", "", "TXT files (*.txt)"
                )
                if txt_file:
                    self.ball_drop_point_timestamp_file = txt_file
                    self.ball_drop_point_video_loaded = True

                # Yeni eklenen kısım: Ball drop point için ikinci txt dosyasını seç
                ball_drop_txt_file, _ = QFileDialog.getOpenFileName(
                    self, "Ball Drop Point TXT dosyasını seçin", "", "TXT files (*.txt)"
                )
                if ball_drop_txt_file:
                    self.load_ball_drop_point_data(ball_drop_txt_file)


    def load_ball_drop_point_data(self, filepath):
        self.ball_drop_point_data = {}
        with open(filepath, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                parts = line.strip().split(',')
                if len(parts) >= 11:
                    frame_n = int(parts[0])
                    timestamp_n = float(parts[1]) / 1e9  # Nanoseconds to seconds
                    x_line = float(parts[8])
                    y_line = float(parts[9])
                    success = parts[6]  # Success değeri
                    self.ball_drop_point_data[frame_n] = (x_line, y_line, timestamp_n, success)


    def play_both_videos(self):
        if not self.player_video_loaded or not self.ball_drop_point_video_loaded:
            QMessageBox.warning(self, "Warning", "Please select both videos.")
            return

        self.total_training_time_label.setText(
            "Total Training Time: 00:00:00\nDate: YYYY-MM-DD"
        )  # Reset the time and date label

        # Start the player video immediately
        self.start_player_video()

        # Delay the ball video by 6000 milliseconds (6 seconds)
        QTimer.singleShot(6000, self.start_ball_drop_video)

        self.is_playing = True
        self.timer.start(30)

    def start_player_video(self):
        if self.selected_svo_player.endswith('.svo'):
            self.svo_video_player = self.process_svo_file(
                self.selected_svo_player, self.player_label, self.svo_video_player
            )
        else:
            self.process_video_file(
                self.selected_svo_player, self.player_label, "player", skip_seconds=16  # Skip first 16 seconds
            )


    def start_ball_drop_video(self):
        if self.selected_svo_ball_drop_point.endswith('.avi'):
            self.process_video_file(
                self.selected_svo_ball_drop_point,
                self.ball_drop_point_label,
                "ball_drop_point",
            )
        else:
            self.svo_video_ball_drop_point = self.process_svo_file(
                self.selected_svo_ball_drop_point,
                self.ball_drop_point_label,
                self.svo_video_ball_drop_point,
                disable_body_tracking=True,
            )


    def play_zed_live(self):
        self.show_camera_selection_dialog()

        if not self.selected_cameras:
            print("Kamera seçimi yapılmadı.")
            return

        for camera_id in self.selected_cameras:
            self.init_zed_camera_for_live(camera_id)

        self.active_view = 'live'
        self.playing = True
        self.timer.start(30)

    def init_zed_camera_for_live(self, camera_id):
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.svo_real_time_mode = True

        self.zed = sl.Camera()
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"ZED kamera başlatılamadı: {err}")
            self.playing = False
            return

        self.zed.enable_positional_tracking(sl.PositionalTrackingParameters())
        self.zed.enable_body_tracking(self.body_param)

    def play_svo(self):
        svo_menu = QMenu(self)
        self.action_play_player = svo_menu.addAction('Oyuncu', self.play_svo_player)
        self.action_play_ball_drop_point = svo_menu.addAction(
            'Top Düşme Noktası', self.play_svo_ball_drop_point
        )
        svo_button_position = self.play_button.mapToGlobal(QPoint(0, 0))
        svo_menu.exec_(svo_button_position)
        self.active_view = 'svo'
        self.playing = True
        self.timer.start(30)

    def process_svo_file(self, svo_file_path, display_label, svo_video, disable_body_tracking=False):
        # Reinitialize the svo_video for each video
        svo_video.close()  # Close any previously opened SVO
        svo_video = sl.Camera()  # Reinitialize to ensure it's separate
        
        init_params = sl.InitParameters()
        init_params.svo_real_time_mode = False
        init_params.set_from_svo_file(svo_file_path)
        svo_video.open(init_params)
        
        svo_video.enable_positional_tracking(sl.PositionalTrackingParameters())
        if not disable_body_tracking:
            svo_video.enable_body_tracking(self.body_param)
        else:
            self.clear_table()

        self.current_display_label = display_label

        svo_image = sl.Mat()
        if svo_video.grab() == sl.ERROR_CODE.SUCCESS:
            svo_video.retrieve_image(svo_image, sl.VIEW.LEFT)
            fps = svo_video.get_camera_information().camera_configuration.fps
            total_frames = svo_video.get_svo_number_of_frames()
            duration = total_frames / fps
            minutes, seconds = divmod(duration, 60)
            milliseconds = (duration - int(duration)) * 1000
            self.total_training_time_label.setText(
                f"Total Training Time: {int(minutes):02}:{int(seconds):02}:{int(milliseconds):03}"
            )

        return svo_video  # Return the initialized camera object

    def process_video_file(self, file_path, display_label, video_type, skip_seconds=0):
        if video_type == "player":
            if self.player_video_cap:
                self.player_video_cap.release()
            self.player_video_cap = cv2.VideoCapture(file_path)

            if self.player_timestamp_file:
                self.load_timestamps(self.player_timestamp_file, "player")

            # Skip the first `skip_seconds` seconds of video
            if skip_seconds > 0:
                fps = self.player_video_cap.get(cv2.CAP_PROP_FPS)
                skip_frames = int(fps * skip_seconds)
                self.player_video_cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

            start_time = time.time()
            self.play_video(self.player_video_cap, display_label, "player", start_time)

        elif video_type == "ball_drop_point":
            if self.ball_drop_point_video_cap:
                self.ball_drop_point_video_cap.release()
            self.ball_drop_point_video_cap = cv2.VideoCapture(file_path)

            if self.ball_drop_point_timestamp_file:
                self.load_timestamps(self.ball_drop_point_timestamp_file, "ball_drop_point")

            start_time = time.time()
            self.play_video(self.ball_drop_point_video_cap, display_label, "ball_drop_point", start_time)


    def clear_table(self):
        self.table_widget.setRowCount(0)

    # Update play_video method to use unified label
    def play_video(self, video_cap, display_label, video_type, start_time):
        if video_cap.isOpened():
            ret, frame = video_cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(
                    rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
                )

                scaled_pixmap = QPixmap.fromImage(convert_to_Qt_format).scaled(
                    display_label.size(), Qt.KeepAspectRatio
                )
                display_label.setPixmap(scaled_pixmap)

                # Update total training time
                elapsed_time = time.time() - start_time
                self.update_total_training_time(elapsed_time)

                # Update video frame and timestamp information
                current_frame = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if video_type == "player":
                    if current_frame in self.player_frame_timestamps:
                        timestamp = self.player_frame_timestamps[current_frame]
                        self.unified_frame_info_label.setText(
                            f"Frame-Timestamp Information\nFrame: {current_frame} | Timestamp: {timestamp:.2f}s"
                        )

                elif video_type == "ball_drop_point":
                    if current_frame in self.ball_drop_point_frame_timestamps:
                        timestamp = self.ball_drop_point_frame_timestamps[current_frame]
                        self.unified_frame_info_label.setText(
                            f"Frame-Timestamp Information\nFrame: {current_frame} | Timestamp: {timestamp:.2f}s"
                        )

                    if current_frame in self.ball_drop_point_data:
                        x_line, y_line, timestamp_n, success = self.ball_drop_point_data[current_frame]
                        self.timestamp_conversion_label.setText(
                            f"Ball Drop Point\nx_line: {x_line:.2f}\ny_line: {y_line:.2f}\nTimestamp: {timestamp:.2f}s"
                        )

                        # Construct the success information string
                        new_success_info = (
                            f"Frame: {current_frame}\n"
                            f"x_Line: {x_line:.2f}\n"
                            f"y_Line: {y_line:.2f}\n"
                            f"Success: {success}\n"
                        )
                        self.update_success_label(new_success_info)
                    else:
                        self.timestamp_conversion_label.setText(
                            f"Ball Drop Point\nNo data for this frame: {current_frame}"
                        )

                # Schedule the next frame update
                QTimer.singleShot(
                    33,  # Approx. 30 FPS
                    lambda: self.play_video(video_cap, display_label, video_type, start_time),
                )


    def resizeEvent(self, event):
        # Example to dynamically resize the video areas while maintaining aspect ratio
        new_width = self.width() // 2
        aspect_ratio = 720 / 540  # Assuming original aspect ratio is 4:3

        player_height = int(new_width / aspect_ratio)
        self.player_label.setFixedSize(new_width, player_height)

        ball_height = int((self.height() - player_height) / 0.95)  # Adjust according to the window height
        self.ball_drop_point_label.setFixedSize(new_width, ball_height)

        super().resizeEvent(event)
            
    def convert_timestamp_to_human_readable(self, timestamp):
        # Ensure timestamp is valid and positive
        if timestamp <= 0:
            return "Invalid timestamp"

        # Convert seconds to a datetime object
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        return dt_object.strftime("%Y-%m-%d %H:%M:%S")

    def update_camera_view(self):
        if not self.playing:
            return

        if self.active_view == 'svo':
            if self.svo_video_player.grab() == sl.ERROR_CODE.SUCCESS:
                image = sl.Mat()
                self.svo_video_player.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
                image_rgb = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2RGB)
                self.update_label_with_image(self.player_label, image_rgb)

                if not self.disable_body_tracking:
                    bodies = sl.Bodies()
                    self.svo_video_player.retrieve_bodies(
                        bodies, self.body_runtime_param
                    )
                    body_array = bodies.body_list
                    if len(body_array) > 0:
                        body = body_array[0]
                        target_indices = [
                            1, 3, 5, 6, 12, 13, 18, 19, 22, 23
                        ]
                        try:
                            k = 0
                            for index_o in target_indices:
                                pitch, yaw, roll = quaternion_to_euler(
                                    body.local_orientation_per_joint[index_o][0],
                                    body.local_orientation_per_joint[index_o][1],
                                    body.local_orientation_per_joint[index_o][2],
                                    body.local_orientation_per_joint[index_o][3],
                                )
                                self.update_table_widget(
                                    k, index_o, pitch, yaw, roll
                                )
                                self.angle_data['pitch'].append(pitch)
                                self.angle_data['yaw'].append(yaw)
                                self.angle_data['roll'].append(roll)
                                k += 1
                        except NameError:
                            print("body.keypoint[0] WASN'T defined!")

                        image_scale = [1.0, 1.0]
                        cv_viewer.render_2D(
                            image_rgb,
                            image_scale,
                            bodies.body_list,
                            self.body_param.enable_tracking,
                            self.body_param.body_format,
                        )

                height, width, channel = image_rgb.shape
                bytes_per_line = 3 * width
                qImg = QImage(
                    image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
                )
                pixmap = QPixmap.fromImage(qImg)
                self.player_label.setPixmap(
                    pixmap.scaled(self.player_label.size(), Qt.KeepAspectRatio)
                )

            if self.svo_video_ball_drop_point.grab() == sl.ERROR_CODE.SUCCESS:
                image = sl.Mat()
                self.svo_video_ball_drop_point.retrieve_image(
                    image, sl.VIEW.LEFT, sl.MEM.CPU
                )
                image_rgb = cv2.cvtColor(image.get_data(), cv2.COLOR_BGR2RGB)
                self.update_label_with_image(self.ball_drop_point_label, image_rgb)
                qImg = QImage(
                    image_rgb.data,
                    image.get_width(),
                    image.get_height(),
                    QImage.Format_RGB888,
                )
                self.ball_drop_point_label.setPixmap(
                    pixmap.scaled(self.ball_drop_point_label.size(), Qt.KeepAspectRatio)
                )
                self.timer.start(30)

                if not self.disable_body_tracking:
                    bodies = sl.Bodies()
                    self.svo_video.retrieve_bodies(
                        bodies, self.body_runtime_param
                    )
                    body_array = bodies.body_list
                    if len(body_array) > 0:
                        body = body_array[0]
                        if len(body_array) > 1:
                            body_2 = body_array[1]
                        image_scale = [1.0, 1.0]
                        cv_viewer.render_2D(
                            image_rgb,
                            image_scale,
                            bodies.body_list,
                            self.body_param.enable_tracking,
                            self.body_param.body_format,
                        )

                        target_indices = [
                            1, 3, 5, 6, 12, 13, 18, 19, 22, 23
                        ]
                        try:
                            k = 0
                            for index_o in target_indices:
                                pitch, yaw, roll = quaternion_to_euler(
                                    body.local_orientation_per_joint[
                                        index_o
                                    ][0],
                                    body.local_orientation_per_joint[
                                        index_o
                                    ][1],
                                    body.local_orientation_per_joint[
                                        index_o
                                    ][2],
                                    body.local_orientation_per_joint[
                                        index_o
                                    ][3],
                                )
                                self.update_table_widget(
                                    k, index_o, pitch, yaw, roll
                                )
                                k = k + 1
                        except NameError:
                            print("body.keypoint[0] WASN'T defined!")

                        height, width, channel = image_rgb.shape
                        bytes_per_line = 3 * width
                        qImg = QImage(
                            image_rgb.data,
                            width,
                            height,
                            bytes_per_line,
                            QImage.Format_RGB888,
                        )
                        pixmap = QPixmap.fromImage(qImg)
                        self.current_display_label.setPixmap(
                            pixmap.scaled(720, 540, Qt.KeepAspectRatio)
                        )

        elif self.active_view == 'live':
            if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                image_2 = sl.Mat()
                self.zed.retrieve_image(image_2, sl.VIEW.LEFT, sl.MEM.CPU)

                bodies = sl.Bodies()
                self.zed.retrieve_bodies(bodies, self.body_runtime_param)

                image_2_ocv = image_2.get_data()
                image_2_rgb = cv2.cvtColor(image_2_ocv, cv2.COLOR_BGR2RGB)

                image_scale = [1.0, 1.0]
                cv_viewer.render_2D(
                    image_2_rgb,
                    image_scale,
                    bodies.body_list,
                    self.body_param.enable_tracking,
                    self.body_param.body_format,
                )

                target_indices = [1, 3, 5, 6, 12, 13, 18, 19, 22, 23]
                if len(bodies.body_list) > 0:
                    body = bodies.body_list[0]
                    try:
                        k = 0
                        for index_o in target_indices:
                            pitch, yaw, roll = quaternion_to_euler(
                                body.local_orientation_per_joint[index_o][0],
                                body.local_orientation_per_joint[index_o][1],
                                body.local_orientation_per_joint[index_o][2],
                                body.local_orientation_per_joint[index_o][3],
                            )
                            self.update_table_widget(k, index_o, pitch, yaw, roll)

                            k += 1
                    except NameError:
                        print("Body keypoints were not defined!")

                    height, width, channel = image_2_rgb.shape
                    bytes_per_line = 3 * width
                    qImg_2 = QImage(
                        image_2_rgb.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888,
                    )
                    pixmap_2 = QPixmap.fromImage(qImg_2)

                    aspect_ratio = width / height
                    new_width = 600
                    new_height = int(new_width / aspect_ratio)

                    self.central_widget.setPixmap(
                        pixmap_2.scaled(new_width, new_height)
                    )



    def update_label_with_image(self, label, image_rgb):
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        qImg = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg).scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(pixmap)
    

    def update_table_widget(self, body_index, joint_index, pitch, yaw, roll):
        target_indices = [1, 3, 5, 6, 12, 13, 18, 19, 22, 23]
        joint_names = {
            1: 'Naval Spine', 3: 'Neck', 5: 'Left Shoulder', 6: 'Left Elbow', 
            12: 'Right Shoulder', 13: 'Right Elbow', 18: 'Left Hip', 19: 'Left Knee',
            22: 'Right Hip', 23: 'Right Knee'
        }

        if joint_index not in target_indices:
            return

        # Tabloya satır ekle
        while self.table_widget.rowCount() <= body_index:
            self.table_widget.insertRow(self.table_widget.rowCount())

        # İlgili bilgileri tabloya ekle
        current_timestamp = time.time()  # Şu anki zamanı al

        joint_name_with_index = f"{joint_index} ({joint_names.get(joint_index, 'Unknown')})"
        item_index = QTableWidgetItem(joint_name_with_index)
        item_pitch = QTableWidgetItem(f"{pitch:.2f}")
        item_yaw = QTableWidgetItem(f"{yaw:.2f}")
        item_roll = QTableWidgetItem(f"{roll:.2f}")

        # Hücrelerin renklerini ayarlama
        item_pitch.setForeground(Qt.red)
        item_yaw.setForeground(QColor(0, 128, 0))  # Daha koyu bir yeşil renk
        item_roll.setForeground(Qt.blue)
        
        # Sütunları ayarla
        self.table_widget.setItem(body_index, 0, item_index)
        self.table_widget.setItem(body_index, 1, item_pitch)
        self.table_widget.setItem(body_index, 2, item_yaw)
        self.table_widget.setItem(body_index, 3, item_roll)

        # Seçilen satırda grafiği güncelle
        if self.selected_row == body_index:
            self.update_graph_with_selected_row(self.selected_row)

        # Eğer bu satır daha önce açılmışsa grafiği güncelle
        if body_index in self.graph_dialogs:
            dialog, canvas, graph_ax = self.graph_dialogs[body_index]
            graph_ax.clear()
            graph_ax.set_xlabel('Açı Türü')
            graph_ax.set_ylabel('Açı (derece)')
            graph_ax.grid(True)
            graph_ax.bar(['Pitch', 'Yaw', 'Roll'], [pitch, yaw, roll], color=['red', 'green', 'blue'])
            canvas.draw()

    def plot_heatmap(self, zoom_factor=1.2, center=None):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from scipy.stats import gaussian_kde

        # Define the dimensions for a half tennis court and extra space
        court_length = 23.77 / 2  # Half of the full length
        court_width = 8.23
        doubles_width = 10.97  # Width including doubles alleys
        extra_space = 4
        service_box_length = court_length / 2

        # Set axis limits to include extra space
        total_length = court_length + extra_space * 2
        total_width = doubles_width + extra_space * 2

        # Clear the axis for fresh plotting
        self.heatmap_ax.clear()

         # Adjust the size of the plot within the fixed label
        self.heatmap_ax.set_position([0.00, 0.00, 1.00, 1.18])  # Adjust position as needed

        # Court lines
        court_start_x = extra_space
        court_start_y = extra_space

        # Draw the tennis court lines
        self.heatmap_ax.plot([court_start_x, court_start_x], [court_start_y, court_start_y + court_width], color="black")  # Left singles sideline
        self.heatmap_ax.plot([court_start_x, court_start_x + court_length], [court_start_y + court_width, court_start_y + court_width], color="black")  # Baseline
        self.heatmap_ax.plot([court_start_x + court_length, court_start_x + court_length], [court_start_y + court_width, court_start_y], color="black")  # Right singles sideline
        self.heatmap_ax.plot([court_start_x + court_length, court_start_x], [court_start_y, court_start_y], color="black")  # Net line

        # Draw the service boxes
        self.heatmap_ax.plot([court_start_x + service_box_length, court_start_x + service_box_length], [court_start_y, court_start_y + court_width], color="black")  # Center service line (only on the right)

        # Draw the center horizontal line at the middle of the right service box
        self.heatmap_ax.plot([court_start_x + service_box_length, court_start_x + court_length], 
                                [court_start_y + court_width / 2, court_start_y + court_width / 2], 
                                color="black")  # Horizontal line in the right service box

        # Add the bottom pink line with the correct alignment and distance
        line_offset = 1.7  # Distance from the bottom of the court (same as the above line's offset)
        self.heatmap_ax.plot([court_start_x, court_start_x + court_length], 
                            [court_start_y - line_offset, court_start_y - line_offset], 
                            color="black")  # Bottom horizontal pink line

        # Add the left vertical extension to align perfectly with the left side of the court
        self.heatmap_ax.plot([court_start_x, court_start_x], 
                            [court_start_y - line_offset, court_start_y + court_width], 
                            color="black")  # Left vertical pink line extension

        # Add the right vertical extension to align perfectly with the right side of the court
        self.heatmap_ax.plot([court_start_x + court_length, court_start_x + court_length], 
                            [court_start_y - line_offset, court_start_y + court_width], 
                            color="black")  # Right vertical pink line extension

        # Re-draw the line outside the lower line (move it downward)
        self.heatmap_ax.plot([court_start_x, court_start_x + court_length], 
                                [court_start_y + court_width + (doubles_width - court_width)/2 + 0.5, 
                                court_start_y + court_width + (doubles_width - court_width)/2 + 0.5], 
                                color="black")  # Bottom doubles alley line (existing)

        self.heatmap_ax.plot([court_start_x, court_start_x], 
                                [court_start_y + (doubles_width - court_width)/2, 
                                court_start_y + court_width + (doubles_width - court_width)/2 + 0.5], 
                                color="black")  # Left doubles sideline
        self.heatmap_ax.plot([court_start_x + court_length, court_start_x + court_length], 
                                [court_start_y + (doubles_width - court_width)/2, 
                                court_start_y + court_width + (doubles_width - court_width)/2 + 0.5], 
                                color="black")  # Right doubles sideline

        # Load data points from CSV
        csv_file = 'C:/Users/Buse/Desktop/arayuz_icin - 240424-1112_T016S3FE-A-001/DusmeNoktasi.txt'
        data = pd.read_csv(csv_file)
        points = list(zip((data['x_line'] / 100) + extra_space, (data['y_line'] / 100) + extra_space))

        # Create a Gaussian kernel density estimate of the distribution
        x, y = np.array(points).T
        kde = gaussian_kde([x, y])

        # Create a grid over the area including the extra space
        xx, yy = np.mgrid[0:total_length:100j, 0:total_width:100j]
        zz = kde(np.vstack([xx.flatten(), yy.flatten()]))

        # Reshape the result and plot it as a heatmap
        img = zz.reshape(xx.shape)
        self.heatmap_ax.imshow(img, extent=(0, total_length, 0, total_width), origin='lower', cmap='YlOrRd', aspect='auto', alpha=0.6)

         # Apply zoom effect
        if center is None:
            center_x, center_y = total_length / 2, total_width / 2
        else:
            center_x, center_y = center

        x_range = total_length / zoom_factor
        y_range = total_width / zoom_factor

        self.heatmap_ax.set_xlim(center_x - x_range / 2, center_x + x_range / 2)
        self.heatmap_ax.set_ylim(center_y - y_range / 2, center_y + y_range / 2)

        self.heatmap_ax.axis('off')

        # Draw the canvas
        self.canvas.draw()

class CameraSelectionDialog(QDialog):
    def __init__(self, camera_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kamera Seçimi")
        self.camera_list = camera_list
        self.selected_cameras = []

        self.list_widget = QListWidget()
        for camera in self.camera_list:
            self.list_widget.addItem(
                f"Kamera ID: {camera.id}, Model: {camera.camera_model}"
            )

        self.select_button = QPushButton("Seç")
        self.select_button.clicked.connect(self.select_cameras)

        layout = QVBoxLayout()
        layout.addWidget(self.list_widget)
        layout.addWidget(self.select_button)
        self.setLayout(layout)

    def select_cameras(self):
        selected_items = self.list_widget.selectedItems()
        self.selected_cameras = [
            self.camera_list[self.list_widget.row(item)].id for item in selected_items
        ]
        self.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = CameraApp()
    mainWindow.show()
    sys.exit(app.exec_())