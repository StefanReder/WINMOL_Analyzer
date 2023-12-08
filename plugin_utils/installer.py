import os
import subprocess
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMessageBox, QProgressDialog
from pkg_resources import find_distributions

WINMOL_VENV_NAME = "winmol_venv"


def get_venv_python_path(venv_path):
    venv_python_path = os.path.join(venv_path, "bin", "python")
    return venv_python_path


def ensure_folder_exists(base_path: Path, folder_name: str) -> Path:
    path = base_path.joinpath(folder_name)
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_requirements_path() -> Path:
    # we assume that a base.txt exists in a requirements folder
    path = Path(
        Path(__file__).parent.parent,
        "requirements",
        "base.txt"
    )
    assert path.exists(), f"path not found {path}"
    return path


def get_tf_requirements_path() -> Path:
    # we assume that a tensorflow.txt exists in a requirements folder
    if sys.platform == "win32":
        tf_requirements_file = "tensorflow-win.txt"
    else:
        tf_requirements_file = "tensorflow.txt"
    path = Path(
        Path(__file__).parent.parent,
        "requirements",
        tf_requirements_file
    )
    assert path.exists(), f"path not found {path}"
    return path


def _dependencies_installed(requirements: str, path: str, is_tf: bool) -> bool:
    if is_tf:
        contains_tf = False
        for d in find_distributions(path):
            print(d, d.key)
            contains_tf = contains_tf or "tensorflow" in d.key
        return contains_tf

    for d in find_distributions(path):
        entry = f"{d.key}=={d.version}"
        print(entry)
        if entry in requirements:
            requirements = requirements.replace(entry, "")

    print(requirements)
    requirements = (requirements
                    .replace(" ", "")
                    .replace(";", "")
                    .replace(",", ""))

    print(requirements)
    if len(requirements) > 0:
        return False
    print("Dependencies already installed")
    return True


def ensure_venv(p, exit_on_miss: bool = False):
    if not os.path.exists(p) and not exit_on_miss:
        message = ("A Python virtual environment (venv) is required update / to"
                   " use the WINMOL_Analyser plugin.\n")
        message += ("Would you like to create / install this venv and the "
                    "required dependencies afterwards?")

        reply = QMessageBox.question(
            None,
            'Missing VENV / Dependencies',
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.No:
            return None

        subprocess.run(
            [
                "python",
                "-m",
                "venv",
                p
            ]
        )
    if not os.path.exists(p) and exit_on_miss:
        raise Exception("Could not resolve venv path.")
    return p


def install_requirements(
        venv_path: str,
        requirements_path: Path
) -> None:
    print(
        "Installing / checking WINMOL_Analyser dependencies in venv {}".format(
            venv_path
        )
    )

    venv_python_path = get_venv_python_path(venv_path)

    print("{} -m pip install -r {}".format(
        venv_python_path,
        requirements_path)
    )

    from subprocess import run
    completed_process = run(
        [
            venv_python_path,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ]
    )

    if completed_process.returncode != 0:
        m = (f"Failed to install dependencies through pip, got "
             f"{completed_process.returncode} as return code. "
             f"Full log: {completed_process}")
        print(m)
        print(completed_process.stdout)
        print(completed_process.stderr)
        raise Exception(m)


def install_dependencies(venv_path: str) -> None:
    print("Check / install base requirements now")
    install_requirements(
        venv_path,
        get_requirements_path()
    )
    print("Check / install base tensorflow requirement now")
    install_requirements(
        venv_path,
        get_tf_requirements_path()
    )

def ensure_dependencies(venv_path: str) -> None:
    try:
        if venv_path is None:
            raise ImportError()
        install_dependencies(venv_path)
        print("Successfully found dependencies")
    except ImportError:
        raise Exception(
            f"Cannot automatically ensure dependencies of WINMOL_Analyser. "
            f"Please try restarting the host application!"
        )
