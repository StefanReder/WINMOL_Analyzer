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

    def run(self):
        self.run_process()
