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
    progress_signal = pyqtSignal(int)
    command = None

    def __init__(self, command):
        super(Worker, self).__init__()
        self.command = command

    def run_process(self):
        total_expected_lines = self.get_total_lines()
        total_lines = 0
        self.progress_signal.emit(0)
        try:
            popen = subprocess.Popen(self.command, stdout=subprocess.PIPE, universal_newlines=True)
            for stdout_line in iter(popen.stdout.readline, ""):
                print(stdout_line)
                total_lines += 1
                self.update_signal.emit(stdout_line)

                # Calculate progress based on total lines
                progress_percentage = (total_lines / total_expected_lines) * 100
                # Update the progress bar
                self.progress_signal.emit(progress_percentage)
            self.progress_signal.emit(100)

            popen.stdout.close()
            return_code = popen.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, self.command)
            self.finished.emit()

        except subprocess.CalledProcessError as e:
            print(str(e))
            self.finished.emit()
            return


    def get_total_lines(self):
        if self.command[-1] == 'Stems':
            return 34
        elif self.command[-1] == 'Trees':
            return 118
        elif self.command[-1] == 'Nodes':
            return 125
