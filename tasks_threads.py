import subprocess

from PyQt5.QtCore import (
    QThread,
    QObject,
    pyqtSignal,
    pyqtSlot,
    QRunnable,
    QEventLoop,
    QThreadPool,
    QSize,
    Qt,
)
import threading
import queue
import time
class Worker(QObject):
    finished = pyqtSignal()
    update_signal = pyqtSignal(str)
    command = None

    def __init__(self, command):
        super(Worker, self).__init__()
        self.command = command

    def run_process(self):
        try:
            process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)

            # Read stdout and stderr in a non-blocking way
            while process.poll() is None:
                # Check if there is data to read from stdout
                stdout_data = process.stdout.readline()
                if stdout_data:
                    print(f"STDOUT: {stdout_data.strip()}")
                    self.update_signal.emit(stdout_data.strip())
             # Wait for the process to complete
            process.wait()

            # Read any remaining output after the process has finished
            remaining_stdout = process.stdout.read()
            if remaining_stdout:
                print(f"STDOUT: {remaining_stdout.strip()}")
                self.update_signal.emit(remaining_stdout.strip())

            self.finished.emit()
        except Exception as e:
            print(f"Error running the process: {e}")
            self.update_signal.emit(f"Error running the process: {e}")
            self.finished.emit()



class PrintStartingTextThread(QThread):

    def __init__(self, model_path, img_path, stem_dir, trees_dir, nodes_dir, process_type, config):
        super(PrintStartingTextThread, self).__init__()
        self.model_path = model_path
        self.img_path = img_path
        self.stem_dir = stem_dir
        self.trees_dir = trees_dir
        self.nodes_dir = nodes_dir
        self.process_type = process_type
        self.config = config
        self.update_text_signal = pyqtSignal(str)

    def run(self):
        self.update_text_signal.emit("Command-line arguments:")
        self.update_text_signal.emit("Model Path: " + self.model_path)
        self.update_text_signal.emit("Image Path: " + self.img_path)
        self.update_text_signal.emit("Semantic Stem Map Directory: " + self.stem_dir)
        if self.trees_dir:
            self.update_text_signal.emit("Detected Wind-thrown Trees Directory: " + self.trees_dir)
        if self.nodes_dir:
            self.update_text_signal.emit("Measuring Nodes Directory: " + self.nodes_dir)
        self.update_text_signal.emit("Process type: " + self.process_type)
        self.update_text_signal.emit("\nConfiguration Settings:")
        self.update_text_signal.emit(self.config.display())


class WorkerSignals(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        """Long-running task."""
        for i in range(5):
            time.sleep(1)
            self.progress.emit(i + 1)
        self.finished.emit()
