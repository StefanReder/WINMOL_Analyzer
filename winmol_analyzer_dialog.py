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
from PyQt5.QtWidgets import QFileDialog
import threading
import queue
#from tensorflow import keras

from PyQt5.QtCore import QThread


from qgis.core import QgsProject, QgsVectorLayer
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

        # parameters
        self.img_path = None
        self.model_path = None
        self.stem_dir = None
        self.trees_dir = None
        self.nodes_dir = None

        # Create a Config instance
        self.config = Config()

        self.set_connections()
        self.output_log.setReadOnly(True)
        self.venv_path = venv_path

    def set_connections(self):
        self.run_button.clicked.connect(self.run_process)
        self.model_comboBox.currentIndexChanged.connect(
            self.handle_model_combo_box_change)
        self.set_parameters()
        self.set_default_config_parameters()
        self.get_config_parameters_from_gui()
        self.uav_toolButton.clicked.connect(self.uav_file_dialog)
        self.model_toolButton.clicked.connect(self.model_file_dialog)
        self.output_toolButton_stem.clicked.connect(self.file_dialog_stem)
        self.output_toolButton_trees.clicked.connect(self.file_dialog_trees)
        self.output_toolButton_nodes.clicked.connect(self.file_dialog_nodes)
        self.output_checkBox_stem.stateChanged.connect(self.checkbox_changed_stem)
        self.output_checkBox_trees.stateChanged.connect(self.checkbox_changed_trees)
        self.output_checkBox_nodes.stateChanged.connect(self.checkbox_changed_nodes)
        self.close_button.clicked.connect(self.close_application)

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

    def set_parameters(self):
        # Extract command-line arguments
        self.img_path = self.uav_lineEdit.text()
        self.model_path = self.model_lineEdit.text()
        self.stem_dir = self.output_lineEdit_stem.text()
        self.trees_dir = self.output_lineEdit_trees.text()
        self.nodes_dir = self.output_lineEdit_nodes.text()

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

    def uav_file_dialog(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TIFF File", "", "TIFF Files (*.tiff *.tif);;All Files (*)", options=options)
        if file_path:
            self.uav_lineEdit.setText(file_path)

    def file_dialog_stem(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Stem Map", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_stem.setText(file_path)

    def file_dialog_trees(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Semantic Stem Map", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_trees.setText(file_path)

    def file_dialog_nodes(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Location And Name For Measuring Nodes", "", "All Files (*)", options=options)
        if file_path:
            self.output_lineEdit_nodes.setText(file_path)

    def checkbox_changed_stem(self, state):
        is_checked = state == 2
        self.output_lineEdit_stem.setEnabled(is_checked)
        self.output_toolButton_stem.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_stem, is_checked)

    def checkbox_changed_trees(self, state):
        is_checked = state == 2
        self.output_lineEdit_trees.setEnabled(is_checked)
        self.output_toolButton_trees.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_trees, is_checked)

    def checkbox_changed_nodes(self, state):
        is_checked = state == 2
        self.output_lineEdit_nodes.setEnabled(is_checked)
        self.output_toolButton_nodes.setEnabled(is_checked)
        self.apply_style_to_line_edit(self.output_lineEdit_nodes, is_checked)

    def apply_style_to_line_edit(self, line_edit, is_checked):
        # Enable or disable the QLineEdit based on the checkbox state
        line_edit.setEnabled(is_checked)

        # Set the stylesheet to gray out the QLineEdit when it is disabled
        if not is_checked:
            line_edit.setStyleSheet("")
        else:
            line_edit.setStyleSheet("QLineEdit { background-color: rgb(255, 255, 255) }")

    def close_application(self):
        print("Closing application")
        self.close()

    def run_process(self):
        # Path to the Python script
        path_dirname = os.path.dirname(__file__)
        script_path = os.path.join(path_dirname, "winmol_run.py")

        model_path = self.model_lineEdit.text()
        img_path = self.uav_lineEdit.text()
        stem_dir = os.path.dirname(self.output_lineEdit_stem.text())
        trees_dir = os.path.dirname(self.output_lineEdit_trees.text())
        nodes_dir = os.path.dirname(self.output_lineEdit_nodes.text())

        if self.output_checkBox_stem.isChecked():
            process_type = "Stems"
        elif self.output_checkBox_trees.isChecked():
            process_type = "Trees"
        elif self.output_checkBox_nodes.isChecked():
            process_type = "Nodes"

        # use python of venv (!)
        command = [
            get_venv_python_path(self.venv_path),
            script_path,
            model_path,
            img_path,
            stem_dir,
            trees_dir,
            process_type
        ]

        # Switch to the log tab in the QTabWidget
        self.log_widget.setCurrentIndex(1)

        self.update_output_log("Starting the process...")

        self.thread = QThread()
        self.worker = Worker(command)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run_process)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.update_signal.connect(self.update_output_log)
        self.thread.start()

    def update_output_log(self, text):
        # Update your QPlainTextEdit with the output
        print("text")
        self.output_log.appendPlainText(text)


#C:/Users/49176/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/standalone/test_stems

#C:/Users/49176/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/standalone/test_trees

#C:/Users/49176/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/WINMOL_Analyzer/standalone/test_nodes
