import os
import shutil
import sys
from importlib import import_module, invalidate_caches
from pathlib import Path
from subprocess import run

from pkg_resources import find_distributions


# Return qgis python path depending on OS
def get_qgis_python_path():
    if sys.platform.startswith("linux"):
        return sys.executable
    python_exec = os.path.dirname(sys.executable)
    if sys.platform == "win32":
        python_exec += "\\python3"
    else:
        python_exec += "/bin/python3"
    return python_exec


def ensure_folder_exists(base_path: Path, folder_name: str) -> Path:
    path = base_path.joinpath(folder_name)
    path.mkdir(exist_ok=True, parents=True)
    return path


print("Starting module dependency installation")
print("Using executable {}".format(sys.executable))
PYTHON_PATH = get_qgis_python_path()


def is_pip_available() -> bool:
    try:
        import_module("pip")  # noqa F401
        return True
    except ImportError:
        return False


def ensure_pip() -> None:
    print("Installing pip... ")
    print(PYTHON_PATH)
    process_cmp = run([PYTHON_PATH, "-m", "ensurepip"])
    if process_cmp.returncode == 0:
        print("Successfully installed pip")
    else:
        raise Exception(
            f"Failed to install pip, got {process_cmp.returncode} return code"
        )


def get_requirements_path() -> Path:
    # we assume that a requirements.txt exists next to the __init__.py file
    path = Path(Path(__file__).parent, "requirements.txt")
    assert path.exists(), f"path not found {path}"
    return path


def _dependencies_installed(requirements: str, path: str) -> bool:
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


def install_requirements(install_path: str) -> None:
    # set up addons/modules under the user
    # script path. Here we'll install the
    # dependencies
    requirements = (get_requirements_path()
                    .read_text().replace("\n", ""))

    if _dependencies_installed(requirements, install_path):
        return

    print("Some requirements are not met. Will reinstall all")
    try:
        shutil.rmtree(install_path)
    except PermissionError:
        raise Exception("Restart QGIS for changes to take effect")

    print(f"Installing WINMOL_Analyser dependencies to {install_path}")

    print("{} -m pip install -t {} -r {}".format(PYTHON_PATH, install_path,
                                                 get_requirements_path()))

    from subprocess import run
    completed_process = run(
        [
            PYTHON_PATH,
            "-m",
            "pip",
            "install",
            "-t",
            str(install_path),
            "-r",
            str(get_requirements_path()),
        ],
        capture_output=True,
    )

    if completed_process.returncode != 0:
        m = (f"Failed to install dependencies through pip, got "
             f"{completed_process.returncode} as return code. "
             f"Full log: {completed_process}")
        print(m)
        print(completed_process.stdout)
        print(completed_process.stderr)
        raise Exception(m)


def install_dependencies(dep_path: str) -> None:
    if not is_pip_available():
        ensure_pip()

    install_requirements(dep_path)


def ensure_dependencies(dep_path: str) -> None:
    try:
        install_dependencies(dep_path)
        invalidate_caches()
        print("Successfully found dependencies")
        if dep_path not in sys.path:
            sys.path.append(dep_path)
            print("Added {} to global path {}", dep_path, sys.path)
    except ImportError:
        raise Exception(
            f"Cannot automatically ensure dependencies of WINMOL_Analyser. "
            f"Please try restarting the host application {dep_path}!"
        )
