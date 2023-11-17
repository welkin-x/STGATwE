# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 18:08:39 2023

@author: ZCX
"""

# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib.pyplot as plt

from PySide2.QtCore import Qt
from PySide2.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QLineEdit, QComboBox, QFrame, 
                               QStackedWidget, QRadioButton, QSizePolicy)


from PySide2.QtGui import QPainter, QLinearGradient, QColor, QFont, QPixmap, QImage
from io import BytesIO
from trial_for_flow_one_month import visualize
from trial_for_flow_one_month import process_data
import datetime
from data_predict import GUI_predict
current_date = datetime.datetime.now()
# 格式化日期为字符串（例如 "2023-03-15"）
formatted_date = current_date.strftime("%Y-%m-%d")
class VisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super(VisualizationWidget, self).__init__(parent)
        self.initUI()
        self.figures = {}

    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        # self.image_label.setMinimumSize(800, 600)  # Make sure the label has a minimum size
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Let the label expand
        self.image_label.setScaledContents(True)  # 确保图像缩放以适应 QLabel 的大小
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)  # 确保使用正确的布局

    def display_figure(self, data_type, visualization_type):
        fig = self.figures.get((data_type, visualization_type))
        if fig:
            print("Displaying figure:", data_type, visualization_type)  # Debug print
            buf = BytesIO()
            fig.savefig(buf, format='png')
            plt.close(fig) 
            buf.seek(0)
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()
            self.image_label.show()  # Ensure the label is visible
            self.image_label.adjustSize()
            self.layout.update()
            self.update()  # Refresh the layout
        else:
            print("Figure not found:", data_type, visualization_type)  # Debug print

    def integrate_visualize_function(self, input_dir):
        visualize(input_dir, self.figures)

    
        

class PredictionWidget(QWidget):
    # Custom QWidget for displaying prediction results
    def __init__(self, parent=None):
        super(PredictionWidget, self).__init__(parent)
        # Initialize your prediction widget layout and styles here
        self.initUI()
        
    def initUI(self):
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        # self.image_label.setMinimumSize(800, 600)  # Make sure the label has a minimum size
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Let the label expand
        self.image_label.setScaledContents(True)  # 确保图像缩放以适应 QLabel 的大小
        self.layout.addWidget(self.image_label)
        self.setLayout(self.layout)  # 确保使用正确的布局

    def display_figure(self, fig_path):
        if fig_path:
            print("Displaying figure:")  # Debug print
            buf = BytesIO()
            fig_path.savefig(buf, format='png')
            plt.close(fig_path) 
            buf.seek(0)
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.adjustSize()
            self.image_label.show()  # Ensure the label is visible
            self.image_label.adjustSize()
            self.layout.update()
            self.update()  # Refresh the layout
        else:
            print("Figure not found")  # Debug print

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Data Visualization and Prediction")
        self.setGeometry(100, 100, 1400, 800)
        self.visualization_widget = VisualizationWidget(self)  # 创建可视化小部件实例
        self.setStyleSheet("QLabel, QPushButton, QComboBox, QLineEdit { color: white; }")
        self.initUI()
    
        
    def initUI(self):
        
        # 设置背景底图
        main_layout = QHBoxLayout()
        
        # Left layout frame and style
        left_frame = QFrame()
        left_layout = QVBoxLayout(left_frame)
        left_frame.setStyleSheet("background-color: rgba(255, 255, 255, 0.05); border-radius: 5px;")
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)
        
        
        # Entrance Path Inputs
        entrance_input_path_label = QLabel("Entrance Input Path:")
        self.entrance_input_path_edit = QLineEdit("F:\AENTRY")  # Entrance path text field
        # Exit Path Inputs
        exit_input_path_label = QLabel("Exit Input Path:")
        self.exit_input_path_edit = QLineEdit("F:\AEXIT")  # Exit path text field
        output_path_label = QLabel("Output Path:")
        self.output_path_edit = QLineEdit("E:\study\STGATwE\data"+formatted_date)  # Output path text field
        read_button = QPushButton("Read")
        read_button.clicked.connect(self.read_data)  # Connect the button click to the read_data method
        
        # Add widgets to the data read layout
        data_read_layout = QVBoxLayout()
        data_read_layout.addWidget(entrance_input_path_label)
        data_read_layout.addWidget(self.entrance_input_path_edit)
        data_read_layout.addWidget(exit_input_path_label)
        data_read_layout.addWidget(self.exit_input_path_edit)
        
        
        data_read_layout.addWidget(output_path_label)
        data_read_layout.addWidget(self.output_path_edit)
        data_read_layout.addWidget(read_button)
        
        # Now add the entire data read layout to the left layout, at the top
        # left_layout.insertLayout(0, data_read_layout)  # The '0' index will insert it at the top
        left_layout.addLayout(data_read_layout)
        # Visualization options
        data_path_layout = QHBoxLayout()
        data_label = QLabel("Data")
        data_label.setStyleSheet("font-weight: bold; color: white;")
        path_label = QLabel("Path: ")
        path_label.setStyleSheet("color: white;")
        self.path_edit = QLineEdit("E:\study\STGATwE\data2023-11-15")
        visualize_button = QPushButton("Visualize")
        visualize_button.clicked.connect(self.visualize_data)
        data_path_layout.addWidget(data_label)
        data_path_layout.addWidget(path_label)
        data_path_layout.addWidget(self.path_edit)
        data_path_layout.addWidget(visualize_button)

        # Entrance and Exit options for visualization
        entrance_exit_layout = QHBoxLayout()
        self.entrance_exit_combo = QComboBox()
        self.entrance_exit_combo.addItems(["Entrance", "Exit"])
        entrance_exit_label = QLabel("Entrance/Exit:")
        entrance_exit_label.setStyleSheet("color: white;")
        entrance_exit_layout.addWidget(entrance_exit_label)
        entrance_exit_layout.addWidget(self.entrance_exit_combo)

        # Visualization options with radio buttons
        visualize_options_layout = QHBoxLayout()
        self.time_radio_btn = QRadioButton("Time")
        self.space_radio_btn = QRadioButton("Space")
        self.od_feature_radio_btn = QRadioButton("OD feature")
        self.correlation_radio_btn = QRadioButton("Correlation")
        self.time_radio_btn.setStyleSheet("color: white;")
        self.space_radio_btn.setStyleSheet("color: white;")
        self.od_feature_radio_btn.setStyleSheet("color: white;")
        self.correlation_radio_btn.setStyleSheet("color: white;")
        self.time_radio_btn.toggled.connect(self.update_visualization)
        self.space_radio_btn.toggled.connect(self.update_visualization)
        self.od_feature_radio_btn.toggled.connect(self.update_visualization)
        self.correlation_radio_btn.toggled.connect(self.update_visualization)
        
        visualize_options_layout.addWidget(self.time_radio_btn)
        visualize_options_layout.addWidget(self.space_radio_btn)
        visualize_options_layout.addWidget(self.od_feature_radio_btn)
        visualize_options_layout.addWidget(self.correlation_radio_btn)
        
        
        
        # Add data path, entrance/exit options, and visualization options to the left layout
        left_layout.addLayout(data_path_layout)
        left_layout.addLayout(entrance_exit_layout)
        left_layout.addLayout(visualize_options_layout)
        
        # Visualization display area
        self.visualization_stack = QStackedWidget()
        self.visualization_widget = VisualizationWidget()
        self.visualization_stack.addWidget(self.visualization_widget)
        self.visualization_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 添加 visualization_stack 到左边的布局中
        left_layout.addWidget(self.visualization_stack)

        # 确保左边的布局也允许内容扩展
        left_frame.setLayout(left_layout)
        
        
        # Right layout frame and style
        right_frame = QFrame()
        right_layout = QVBoxLayout(right_frame)
        right_frame.setStyleSheet("background-color: rgba(255, 255, 255, 0.05); border-radius: 5px;")
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(10)

        # Model selection part
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Model selected: ")
        model_label.setStyleSheet("color: white;")
        train_label = QLabel("Whether to train:")
        train_label.setStyleSheet("color: white;")
        
        self.model_combo_box = QComboBox()
        self.model_combo_box.addItems(["STGAT", "STGAT_noedge","STGCN", "GCN", "GLU","sim_STGAT","LSTM","MLP"])
        
        self.if_train = QComboBox()
        self.if_train.addItems(["NO", "YES"])
        
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.model_combo_box)
        model_selection_layout.addWidget(train_label)
        model_selection_layout.addWidget(self.if_train)
        model_selection_layout.addWidget(self.predict_button)
 
        # predict Visual display area 1 
        self.Prediction_visual_stack = QStackedWidget()
        self.Prediction_visual_widget = PredictionWidget()
        self.Prediction_visual_stack.addWidget(self.Prediction_visual_widget)
        self.Prediction_visual_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 添加 Prediction_visual_stack 到右边的布局中
        right_layout.addWidget(self.Prediction_visual_stack)
        
        
        # predict Visual display area 2 
        self.Prediction_visual_stack_2 = QStackedWidget()
        self.Prediction_visual_widget_2 = PredictionWidget()
        self.Prediction_visual_stack_2.addWidget(self.Prediction_visual_widget_2)
        self.Prediction_visual_stack_2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # 添加 Prediction_visual_stack 到右边的布局中
        right_layout.addWidget(self.Prediction_visual_stack_2)
        # 确保右边的布局也允许内容扩展
        right_frame.setLayout(right_layout)
        
        # Evaluation metrics display, with adjusted height
        metrics_layout = QHBoxLayout()
        mae_label = QLabel("MAE:")
        self.mae_value_label = QLabel("0.00")  # Placeholder for MAE value
        rmse_label = QLabel("RMSE:")
        self.rmse_value_label = QLabel("0.00")  # Placeholder for RMSE value
        r2_label = QLabel("R^2:")
        self.r2_value_label = QLabel("0.00")  # Placeholder for R^2 value
        
        # Style for metrics labels
        metrics_label_style = "font-weight: bold; color: white;"
        mae_label.setStyleSheet(metrics_label_style)
        rmse_label.setStyleSheet(metrics_label_style)
        r2_label.setStyleSheet(metrics_label_style)
        
        # Style for metrics value labels with smaller font
        metrics_value_label_style = "color: white; font-size: 14px;"
        self.mae_value_label.setStyleSheet(metrics_value_label_style)
        self.rmse_value_label.setStyleSheet(metrics_value_label_style)
        self.r2_value_label.setStyleSheet(metrics_value_label_style)
        
        # Adjusting layout spacing to make the metrics display more compact
        metrics_layout.addWidget(mae_label)
        metrics_layout.addWidget(self.mae_value_label)
        metrics_layout.addWidget(rmse_label)
        metrics_layout.addWidget(self.rmse_value_label)
        metrics_layout.addWidget(r2_label)
        metrics_layout.addWidget(self.r2_value_label)
        metrics_layout.setSpacing(5)  # Reduce spacing between widgets
        
        
        
        
        right_layout.addLayout(model_selection_layout)
        # right_layout.addLayout(prediction_results_layout)
        right_layout.addLayout(metrics_layout)

        

        # Add left and right layouts to the main layout
        main_layout.addWidget(left_frame, 1)  # Left frame ratio set to 1
        main_layout.addWidget(right_frame, 1)  # Right frame ratio set to 1

        # Set central widget and apply main layout
        
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor('#2c2c2c'))  # Dark grey
        gradient.setColorAt(1.0, QColor('#191919'))  # Even darker grey
        painter.fillRect(self.rect(), gradient)
    
    def read_data(self):
        # 从 QLineEdit 控件获取输入和输出路径
        entrance_input_path = self.entrance_input_path_edit.text()
        exit_input_path = self.exit_input_path_edit.text()
        output_path = self.output_path_edit.text()

        # 调用 process_data 函数处理数据
        try:
            process_data(entrance_input_path, exit_input_path, output_path)
            print(f"Data processed and written to {output_path}")
        except Exception as e:
            print(f"Error during data processing: {e}")
    def visualize_data(self):
        # Get the path from QLineEdit widget
        input_dir = self.path_edit.text()
        self.visualization_widget.integrate_visualize_function(input_dir)
        self.update_visualization()

    def update_visualization(self):
        entrance_or_exit = 'entrance'  # Default selection, you can modify as needed
        visual_type = 'time'  # Default selection, you can modify as needed
        print(1)
        # Update to get current selection from the GUI
        if self.entrance_exit_combo.currentText():
            entrance_or_exit = self.entrance_exit_combo.currentText().lower()
        
        if self.time_radio_btn.isChecked():
            visual_type = 'time'
        elif self.space_radio_btn.isChecked():
            visual_type = 'space'
        elif self.od_feature_radio_btn.isChecked():
            visual_type = 'OD feature'
        elif self.correlation_radio_btn.isChecked():
            visual_type = 'correlation'
        
        # Display the selected figure
        self.visualization_widget.display_figure(entrance_or_exit, visual_type)
        
    def predict(self):
        # 获取输入目录和模型种类
        input_dir = self.path_edit.text()
        model_kind = self.model_combo_box.currentText()
        if_train = self.if_train.currentText()
        
        # 调用预测接口函数
        fig_en, fig_ex, mae, rmse, r2 = GUI_predict(input_dir, model_kind, if_train)
        
        
        self.mae_value_label.setText(str(round(mae,3)))
        self.rmse_value_label.setText(str(round(rmse,3)))
        self.r2_value_label.setText(str(round(r2,3)))
        # 显示图像和指标
        self.Prediction_visual_widget.display_figure(fig_en)
        self.Prediction_visual_widget_2.display_figure(fig_ex)
        
        
        
        
        
# Main program entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Arial", 14)
    app.setFont(font)
    window = MainWindow()
    window.show()
    app.exec_()
        