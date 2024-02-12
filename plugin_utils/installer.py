import os
import subprocess
import sys
from pathlib import Path
import urllib.request
import json
import shutil

from pkg_resources import find_distributions
from PyQt5.QtWidgets import QMessageBox

WINMOL_VENV_NAME = "winmol_venv"
MODELS_PATH = "models"


def get_python_command():
    python_command = "python3" if shutil.which("python3") else "python"
    return python_command


def get_venv_python_path(venv_path):
    if sys.platform == "win32":
        venv_python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_command = get_python_command()
        venv_python_path = os.path.join(venv_path, "bin", python_command)
    return venv_python_path


def ensure_folder_exists(base_path: Path, folder_name: str) -> Path:
    path = base_path.joinpath(folder_name)
    path.mkdir(exist_ok=True, parents=True)
    return path


def get_requirements_path() -> Path:
    # we assume that a base.txt exists in a requirements folder
    path = Path(Path(__file__).parent.parent, "requirements", "base.txt")
    assert path.exists(), f"path not found {path}"
    return path


def get_tf_requirements_path() -> Path:
    # we assume that a tensorflow.txt exists in a requirements folder
    if sys.platform == "win32":
        tf_requirements_file = "tensorflow-win.txt"
    else:
        tf_requirements_file = "tensorflow.txt"
    path = Path(
        Path(__file__).parent.parent, "requirements", tf_requirements_file
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
    requirements = (
        requirements.replace(" ", "").replace(";", "").replace(",", "")
    )

    print(requirements)
    if len(requirements) > 0:
        return False
    print("Dependencies already installed")
    return True


def ensure_venv(p, exit_on_miss: bool = False):
    if not os.path.exists(p) and not exit_on_miss:
        message = (
            "A Python virtual environment (venv) as well as models"
            " data are required to use the WINMOL_Analyser"
            " plugin.\n"
        )
        message += (
            "Would you like to create / install these " "required dependencies?"
        )

        reply = QMessageBox.question(
            None,
            "Missing VENV / Dependencies",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.No:
            return None

        python_command = get_python_command()
        subprocess.run([python_command, "-m", "venv", "--copies", p])
    if not os.path.exists(p) and exit_on_miss:
        raise Exception("Could not resolve venv path.")
    return p


def ensure_pip(venv_path) -> None:
    print("Installing pip... ")
    try:
        process_cmp = subprocess.run(
            [get_venv_python_path(venv_path), "-m", "ensurepip"]
        )
        if process_cmp.returncode == 0:
            print("Successfully installed pip")
    except subprocess.CalledProcessError as e:
        raise Exception(f"error: {e.stderr}")


def install_requirements(venv_path: str, requirements_path: Path) -> None:
    print(
        "Installing / checking WINMOL_Analyser dependencies in venv {}".format(
            venv_path
        )
    )

    venv_python_path = get_venv_python_path(venv_path)

    print("{} -m pip install -r {}".format(venv_python_path, requirements_path))

    completed_process = subprocess.run(
        [
            venv_python_path,
            "-m",
            "pip",
            "install",
            "-r",
            str(requirements_path),
        ],
        capture_output=True,
    )

    if completed_process.returncode != 0:
        m = (
            f"Failed to install dependencies through pip, got "
            f"{completed_process.returncode} as return code. "
            f"Full log: {completed_process}"
        )
        print(m)
        print(completed_process.stdout)
        print(completed_process.stderr)
        raise Exception(m)


def install_dependencies(venv_path: str) -> None:
    ensure_pip(venv_path)
    print("Check / install base requirements now")
    install_requirements(venv_path, get_requirements_path())
    print("Check / install base tensorflow requirement now")
    install_requirements(venv_path, get_tf_requirements_path())


def ensure_dependencies(venv_path: str) -> None:
    try:
        if venv_path is None:
            raise ImportError()
        install_dependencies(venv_path)
        download_models(venv_path)
        print("Successfully found dependencies")
    except ImportError:
        raise Exception(
            "Cannot automatically ensure dependencies of WINMOL_Analyser. "
            "Please try restarting QGIS and / or reinstalling the plugin."
        )


def download_models(venv_path: str):
    # get project path from venv path
    project_path = os.path.dirname(venv_path)
    download_directory = os.path.join(project_path, MODELS_PATH)
    if not os.path.exists(download_directory):
        os.makedirs(download_directory)
    try:
        with open(os.path.join(project_path, "config.json")) as f:
            files = json.load(f)

        for file_name, url in files.items():
            file_path = os.path.join(download_directory, f"{file_name}.hdf5")
            # check if file exists
            if not os.path.exists(file_path):
                print(f"{file_name} data not found, downloading...")
                urllib.request.urlretrieve(url, file_path)
                print(f"Downloaded {file_name} data.")
            else:
                print(f"{file_name} data exists.")

    # except download error
    except urllib.error.URLError as e:
        print(e.reason)
        raise Exception(
            "Cannot automatically ensure models of WINMOL_Analyser. "
            "Please try restarting QGIS and / or reinstalling the plugin."
        )
