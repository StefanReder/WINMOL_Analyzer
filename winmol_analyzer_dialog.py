# -*- coding: utf-8 -*-
"""
/***************************************************************************
 WINMOLAnalyzer
Dialog
                                 A QGIS plugin
 Plugin to detect stems from UAV
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2023-11-29
        git sha              : $Format:%H$
        copyright            : (C) 2023 by Hochschule für nachhaltige Entwicklung Eberswalde | mundialis GmbH & Co. KG | terrestris GmbH & Co. KG
        email                : Stefan.Reder@hnee.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os
import subprocess

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import threading
import queue
#from tensorflow import keras

from PyQt5.QtCore import QThread


from qgis.core import QgsProject, QgsVectorLayer, QgsRasterLayer, QgsCoordinateReferenceSystem
from qgis.PyQt import QtWidgets, uic

from .classes.Config import Config
from .plugin_utils.installer import get_venv_python_path
from .tasks_threads import Worker

current_path = os.path.dirname(__file__)

# This loads your .ui file so that PyQt can populate your plugin with the
# elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(
    os.path.join(current_path, 'winmol_analyzer_dialog_base.ui')
)


class WINMOLAnalyzerDialog(QtWidgets.QDialog, FORM_CLASS):

    def __init__(self, parent=None, venv_path=None):
        """Constructor."""
        super(WINMOLAnalyzerDialog, self).__init__(parent)
        self.setupUi(self)

        # Set the initial title
        self.setWindowTitle('WINMOL Analyzer')

        # parameters
        self.uav_path = ''
        self.model_path = ''
        self.stem_path = ''
        self.trees_path = ''
        self.nodes_path = ''
        self.process_type = 'Stems'

        self.uav_layer_path = None
        self.uav_layer_name = None
        self.crs = None
        self.worker = None
        self.thread = None

        # Create a Config instance
        self.config = Config()

        self.set_connections()
        self.output_log.setReadOnly(True)
        self.venv_path = venv_path
        self.process_type = None

        # hide warning label
        self.uav_warning_label.hide()

    def set_connections(self):
        self.run_button.clicked.connect(self.run_process)
        self.model_comboBox.currentIndexChanged.connect(
            self.handle_model_combo_box_change)
        self.set_default_config_parameters()
        self.get_config_parameters_from_gui()
        self.uav_toolButton.clicked.connect(self.file_dialog_uav)
        self.model_toolButton.clicked.connect(self.model_file_dialog)
        self.output_toolButton_stem.clicked.connect(self.file_dialog_stem)
        self.output_toolButton_trees.clicked.connect(self.file_dialog_trees)
        self.output_toolButton_nodes.clicked.connect(self.file_dialog_nodes)
        self.output_checkBox_stem.stateChanged.connect(self.checkbox_changed_stem)
        self.output_checkBox_trees.stateChanged.connect(self.checkbox_changed_trees)
        self.output_checkBox_nodes.stateChanged.connect(self.checkbox_changed_nodes)
        self.checkbox_changed_stem(2)
        self.checkbox_changed_trees(1)
        self.checkbox_changed_nodes(1)
        self.close_button.clicked.connect(self.close_application)
        self.cancel_button.clicked.connect(self.cancel_process)

    def handle_model_combo_box_change(self):
        selected_text = self.model_comboBox.currentText()
        widgets_to_enable = [
            self.tileside_label, self.image_spinBox, self.model_lineEdit,
            self.model_toolButton, self.segm_label, self.tileside_doubleSpinBox,
            self.tileside_unit_label, self.image_label, self.image_spinBox,
            self.image_unit_label
        ]

        for widget in widgets_to_enable:
            widget.setEnabled(selected_text == "Custom")

        if selected_text == "Custom":
            self.apply_style_to_line_edit(self.model_lineEdit, True)
        else:
            self.apply_style_to_line_edit(self.model_lineEdit, False)

    def model_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model File (*.hdf5);;All Files (*)", options=options)
        if file_path:
            self.model_lineEdit.setText(file_path)
        self.model_path = self.model_lineEdit.text()

    def set_default_config_parameters(self):
        # set default values
        self.minlength_doubleSpinBox.setValue(self.config.min_length)
        self.maxdistance_doubleSpinBox.setValue(self.config.max_distance)
        self.tolerance_doubleSpinBox.setValue(self.config.tolerance_angle)
        self.maxtree_doubleSpinBox.setValue(self.config.max_tree_height)
        self.tileside_doubleSpinBox.setValue(self.config.tile_size)
        self.image_spinBox.setValue(self.config.img_width)

    def get_config_parameters_from_gui(self):
        self.config.min_length = self.minlength_doubleSpinBox.value()
        self.config.max_distance = self.maxdistance_doubleSpinBox.value()
        self.config.tolerance_angle = self.tolerance_doubleSpinBox.value()
        self.config.max_tree_height = self.maxtree_doubleSpinBox.value()

    def set_crs(self, layer):
        # get crs from uav image
        crs = layer.crs()
        # set crs to config
        self.crs = crs

    def check_uav_pixel_size(self):
        # get file name of uav image
        file_name = os.path.splitext(os.path.basename(self.uav_path))[0]
        # Load the raster layer
        layer = QgsRasterLayer(self.uav_lineEdit.text(), file_name, "gdal")

        # Check if the layer is valid
        if layer.isValid():

            # set crs
            self.set_crs(layer)

            # Get the raster's pixel size in map units (usually meters)
            x_pixel_size = layer.rasterUnitsPerPixelX()
            y_pixel_size = layer.rasterUnitsPerPixelY()

            # Convert pixel size to centimeters
            x_pixel_size_cm = x_pixel_size * 100
            y_pixel_size_cm = y_pixel_size * 100

            # Set the threshold for showing the warning label (5 cm)
            threshold_cm = 5

            # Check if either the x or y pixel size exceeds the threshold
            if x_pixel_size_cm > threshold_cm or y_pixel_size_cm > threshold_cm:
                # show warning label
                self.uav_warning_label.show()
            else:
                # hide warning label
                self.uav_warning_label.hide()
        else:
            print('Invalid raster layer. Check the path and format.')

    def file_dialog_uav(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TIFF File", "", "TIFF Files (*.tiff *.tif);;All Files (*)", options=options)
        if file_path:
            self.uav_lineEdit.setText(file_path)
        # check if pixel size is too large
        self.check_uav_pixel_size()
        self.uav_path = self.uav_lineEdit.text()

    def file_dialog_stem(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Stem Map", "", "All Files (*)", options=options)
        if file_path:
            # Check if the file is already loaded in QGIS
            self.check_uav_input_exists(file_path)
            # File is not loaded in QGIS or user chose to remove, set the text in the line edit
            self.output_lineEdit_stem.setText(file_path)
            self.stem_path = self.output_lineEdit_stem.text()

    def check_uav_input_exists(self, file_path):
        # Check if the file is already loaded in QGIS
        loaded_layers = QgsProject.instance().mapLayers().values()
        if any(layer.source() == file_path for layer in loaded_layers):
            matching_layers = [layer for layer in loaded_layers if layer.source() == file_path]
            # Remove the matching layers
            for layer in matching_layers:
                QgsProject.instance().removeMapLayer(layer.id())

    def file_dialog_trees(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Semantic Stem Map", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_trees.setText(file_path)
            self.trees_path = self.output_lineEdit_trees.text()

    def file_dialog_nodes(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Measuring Nodes", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_nodes.setText(file_path)
            self.nodes_path = self.output_lineEdit_nodes.text()

    def checkbox_changed_stem(self, state):
        is_checked = state == 2
        self.output_lineEdit_stem.setEnabled(is_checked)
        self.output_toolButton_stem.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_stem, is_checked)
        self.output_checkBox_trees.setEnabled(is_checked)

    def checkbox_changed_trees(self, state):
        is_checked = state == 2
        self.output_lineEdit_trees.setEnabled(is_checked)
        self.output_toolButton_trees.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_trees, is_checked)
        self.output_checkBox_nodes.setEnabled(is_checked)

    def checkbox_changed_nodes(self, state):
        is_checked = state == 2
        self.output_lineEdit_nodes.setEnabled(is_checked)
        self.output_toolButton_nodes.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_nodes, is_checked)

    def set_selected_process_type(self):
        stem_checked = self.output_checkBox_stem.isChecked()
        trees_checked = self.output_checkBox_trees.isChecked()
        nodes_checked = self.output_checkBox_nodes.isChecked()

        if nodes_checked:
            self.process_type = 'Nodes'
        elif trees_checked:
            self.process_type = 'Trees'
        elif stem_checked:
            self.process_type = 'Stems'
        else:
            self.process_type = 'Stems'

    def save_temp_layer(self, layer, layer_name):
        layer = self.uav_layer_path
        # Save the layer to the project as temporary layer
        QgsProject.instance().addMapLayer(layer, False)
        # Set the layer name
        layer.setName(layer_name)

    def apply_style_to_line_edit(self, line_edit, is_checked):
        # Enable or disable the QLineEdit based on the checkbox state
        line_edit.setEnabled(is_checked)

        # Set the stylesheet to gray out the QLineEdit when it is disabled
        if not is_checked:
            line_edit.setStyleSheet("")
        else:
            line_edit.setStyleSheet("QLineEdit { background-color: rgb(255, 255, 255) }")

    def cancel_process(self):
        # If the process is running, cancel it
        if self.worker:
            self.worker.finished.emit()
            self.thread.quit()

    def close_application(self):
        print("Closing application")
        self.close()


    def run_process(self):
        # Path to the Python script
        path_dirname = os.path.dirname(__file__)
        script_path = os.path.join(path_dirname, "winmol_run.py")


        # check process type from checkboxes
        self.set_selected_process_type()

        # check if uav image is loaded in qgis
        self.check_uav_input_exists(self.stem_path)

        # use python of venv (!)
        command = [
            get_venv_python_path(self.venv_path),
            '-u',
            script_path,
            self.model_path,
            self.uav_path,
            self.stem_path,
            self.trees_path,
            self.nodes_path,
            self.process_type
        ]

        # Switch to the log tab in the QTabWidget
        self.log_widget.setCurrentIndex(1)

        # clear the output log
        self.output_log.clear()

        self.update_output_log("Starting the process...")


        # for debugging run subprocess directly
        # process = subprocess.run(
        #     command,
        #     capture_output=True,
        #     text=True
        # )
        # print(process.stdout)
        # print(process.stderr)
        # self.update_output_log(process.stdout)
        # self.update_output_log(process.stderr)


        # Run this part for responcive GUI
        self.thread = QThread()
        self.worker = Worker(command)
        self.worker.moveToThread(self.thread)
        self.worker.progress_signal.connect(self.update_progress)
        self.thread.started.connect(self.worker.run_process)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.load_layers_to_session)
        self.worker.update_signal.connect(self.update_output_log)
        self.thread.start()

    def update_output_log(self, text):
        # Update your QPlainTextEdit with the output
        self.output_log.appendPlainText(text)

    def load_layers_to_session(self):
        self.load_raster(self.stem_path)
        if self.trees_path:
            self.load_geojson(self.trees_path + '_stems.geojson')
        if self.nodes_path:
            self.load_geojson(self.nodes_path + '_vectors.geojson')
            self.load_geojson(self.nodes_path + '_nodes.geojson')

    def load_raster(self, path):
        # Extract the base name of the input image file
        name = os.path.splitext(os.path.basename(path))[0]
        # Load raster layer
        raster_layer = QgsRasterLayer(path, name)
        QgsProject.instance().addMapLayer(raster_layer)


    def load_geojson(self, path):
        # Extract the base name of the input image file
        name = os.path.splitext(os.path.basename(path))[0]
        # Load vector layer
        vector_layer = QgsVectorLayer(path, name, "ogr")
        if not vector_layer.isValid():
            print(f"Error loading vector layer from {path}")
        else:
            # Add the vector layer to the map
            QgsProject.instance().addMapLayer(vector_layer)

    def update_progress(self, value):
        self.progress_bar.setValue(value)
