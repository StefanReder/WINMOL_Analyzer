import os
import shutil
import sys
from importlib import import_module, invalidate_caches
from pathlib import Path
from subprocess import run

from pkg_resources import find_distributions


class Installer:
    PYTHON_PATH = None

    def __init__(self):
        print("Starting module dependency installation")
        print("Using executable {}".format(sys.executable))
        self.PYTHON_PATH = self.get_qgis_python_path()

    # Return qgis python path depending on OS
    def get_qgis_python_path(self):
        if sys.platform.startswith("linux"):
            return sys.executable
        python_exec = os.path.dirname(sys.executable)
        if sys.platform == "win32":
            python_exec += "\\python3"
        else:
            python_exec += "/bin/python3"
        return python_exec

    def ensure_folder_exists(self, base_path: Path, folder_name: str) -> Path:
        path = base_path.joinpath(folder_name)
        path.mkdir(exist_ok=True, parents=True)
        return path

    def is_pip_available(self) -> bool:
        try:
            import_module("pip")  # noqa F401
            return True
        except ImportError:
            return False

    def ensure_pip(self) -> None:
        print("Installing pip... ")
        print(self.PYTHON_PATH)
        process_cmp = run([self.PYTHON_PATH, "-m", "ensurepip"])
        if process_cmp.returncode == 0:
            print("Successfully installed pip")
        else:
            raise Exception(
                f"Failed to install pip, got {process_cmp.returncode}"
            )

    def get_requirements_path(self) -> Path:
        # we assume that a base.txt exists in a requirements folder
        path = Path(
            Path(__file__).parent.parent,
            "requirements",
            "base.txt"
        )
        assert path.exists(), f"path not found {path}"
        return path

    def get_tf_requirements_path(self) -> Path:
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

    def _dependencies_installed(self, requirements: str, path: str) -> bool:
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

    def ensure_dependency_dir(self, p, exit_on_miss: bool = False):
        if not os.path.exists(p) and not exit_on_miss:
            os.makedirs(p)
        if not os.path.exists(p) and exit_on_miss:
            raise Exception("Could not resolve dependency path.")
        return p

    def install_requirements(
            self,
            install_path: str,
            requirements_path: Path,
            is_tf: bool
    ) -> None:
        # set up addons/modules under the user
        # script path. Here we'll install the
        # dependencies
        requirements = (requirements_path
                        .read_text().replace("\n", "")).lower()

        if not sys.platform == "win32" and is_tf:
            requirements = (requirements.replace("[and-cuda]", ""))

        if self._dependencies_installed(requirements, install_path):
            return

        if not is_tf:
            print("Some requirements are not met. Will reinstall all")
            try:
                shutil.rmtree(install_path)
                self.ensure_dependency_dir(install_path)
            except PermissionError:
                raise Exception("Restart QGIS for changes to take effect")

        print(f"Installing WINMOL_Analyser dependencies to {install_path}")

        print("{} -m pip install -t {} -r {}".format(
            self.PYTHON_PATH,
            install_path,
            requirements_path)
        )

        from subprocess import run
        completed_process = run(
            [
                self.PYTHON_PATH,
                "-m",
                "pip",
                "install",
                "-t",
                str(install_path),
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

    def install_dependencies(self, dep_path: str) -> None:
        if not self.is_pip_available():
            self.ensure_pip()

        print("Check / install base requirements now")
        self.install_requirements(
            dep_path,
            self.get_requirements_path(),
            False
        )

        print("Check / install base tensorflow requirement now")
        self.install_requirements(
            dep_path,
            self.get_tf_requirements_path(),
            True
        )

    def set_path(self, dep_path) -> None:
        print("Setting Path")
        # for d in find_distributions(dep_path):
        #     lib_path = "{}/{}".format(dep_path, d.key)
        #     if lib_path not in sys.path:
        #         sys.path.append(lib_path)
        #         print("Added {} to global path", lib_path)
        sys.path.append(dep_path)
        print("global path {}", sys.path)

    def ensure_dependencies(self, dep_path: str) -> None:
        try:
            self.install_dependencies(dep_path)
            invalidate_caches()
            self.set_path(dep_path)
            print("Successfully found dependencies")
        except ImportError:
            raise Exception(
                f"Cannot automatically ensure dependencies of WINMOL_Analyser. "
                f"Please try restarting the host application {dep_path}!"
            )
