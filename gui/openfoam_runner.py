"""
OpenFOAM/WSL command-building for the GUI.

Split out of gui/main.py's RunPage so the bash/WSL glue code (building
shell commands, staging paths for WSL, writing temp scripts) isn't mixed
in with Tkinter widget code. Nothing in this module touches Tk or GUI
state directly -- callers pass in the project/output folders explicitly.
"""

import os
import shutil
import subprocess


class OpenFoamRunner:
    """Builds the WSL/bash commands used to drive OpenFOAM mesh builds."""

    def __init__(self, project_folder):
        self.project_folder = project_folder

    # -- path helpers --------------------------------------------------

    def get_passage_meshes_folder(self):
        return os.path.join(self.project_folder, "passageMeshes")

    def candidate_bash_script_paths(self):
        return [
            os.path.join(self.project_folder, "runtest.sh"),
            os.path.join(self.project_folder, "passageMeshes", "runtest.sh"),
            os.path.join(self.project_folder, "passageMeshes", "multipassagetest.sh"),
        ]

    def find_bash_script(self):
        for candidate in self.candidate_bash_script_paths():
            if os.path.exists(candidate):
                return os.path.abspath(candidate)

        return None

    def find_multipassage_script(self):
        candidate = os.path.join(self.project_folder, "passageMeshes", "multipassagetest.sh")
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        return None

    def get_bash_working_folder(self, bash_script):
        script_name = os.path.basename(bash_script)

        if script_name == "multipassagetest.sh":
            return os.path.dirname(bash_script)

        return self.project_folder

    def windows_to_wsl_path(self, windows_path):
        try:
            completed = subprocess.run(
                ["wsl", "wslpath", "-a", windows_path],
                capture_output=True,
                text=True,
                check=True
            )
            return completed.stdout.strip()
        except Exception:
            drive, path_tail = os.path.splitdrive(os.path.abspath(windows_path))
            drive_letter = drive.replace(":", "").lower()
            path_tail = path_tail.replace("\\", "/")
            return f"/mnt/{drive_letter}{path_tail}"

    @staticmethod
    def is_passage_folder_name(folder_name):
        return folder_name.startswith("passage") and folder_name[7:].isdigit()

    @staticmethod
    def passage_sort_key(folder_name):
        try:
            return int(folder_name[7:])
        except ValueError:
            return 0

    def find_generated_passages(self, output_folder):
        if not os.path.isdir(output_folder):
            return []

        passages = []
        for folder_name in os.listdir(output_folder):
            folder_path = os.path.join(output_folder, folder_name)
            if os.path.isdir(folder_path) and self.is_passage_folder_name(folder_name):
                passages.append(folder_name)

        return sorted(passages, key=self.passage_sort_key)

    @staticmethod
    def shell_quote(value):
        return "'" + str(value).replace("'", "'\\''") + "'"

    @staticmethod
    def posix_join(*parts):
        cleaned_parts = []

        for index, part in enumerate(parts):
            part = str(part).replace("\\", "/")

            if index == 0:
                cleaned_parts.append(part.rstrip("/"))
            else:
                cleaned_parts.append(part.strip("/"))

        return "/".join(cleaned_parts)

    @staticmethod
    def openfoam_source_command():
        return (
            "if [ -f /opt/openfoam12/etc/bashrc ]; then . /opt/openfoam12/etc/bashrc; "
            "elif [ -f /usr/lib/openfoam/openfoam12/etc/bashrc ]; then . /usr/lib/openfoam/openfoam12/etc/bashrc; "
            "else true; fi"
        )

    # -- shell command templates ----------------------------------------

    def single_passage_shell_command(self, project_folder, output_folder, passage_name="passage0"):
        passage_meshes_folder = self.posix_join(project_folder, "passageMeshes")

        return " && ".join([
            self.openfoam_source_command(),
            f"cd {self.shell_quote(passage_meshes_folder)}",
            f"rm -rf {self.shell_quote(passage_name)}",
            f"mkdir -p {self.shell_quote(passage_name)}",
            f"cp -r template/* {self.shell_quote(passage_name)}/",
            f"mkdir -p {self.shell_quote(passage_name)}/constant/geometry",
            f"mkdir -p {self.shell_quote(passage_name)}/system",
            f"cp {self.shell_quote(self.posix_join(output_folder, passage_name))}/*.stl {self.shell_quote(passage_name)}/constant/geometry/",
            f"cp {self.shell_quote(self.posix_join(output_folder, passage_name, 'passageParameters'))} {self.shell_quote(passage_name)}/system/",
            f"cd {self.shell_quote(passage_name)}",
            "sed -i 's/\\r$//' system/geomUpdate.sh 2>/dev/null || true",
            "sed -i 's/#eval{\\([^}]*\\)}/#calc \"\\1\"/g' system/blockMeshDict 2>/dev/null || true",
            "bash ./geomUpdate.sh",
            "sed -i 's/#eval{\\([^}]*\\)}/#calc \"\\1\"/g' system/blockMeshDict 2>/dev/null || true",
            "blockMesh",
            "checkMesh",
        ])

    def wsl_single_passage_shell_command(self, source_passage_meshes_folder, source_output_folder, passage_name="passage0"):
        source_passage_meshes_folder = source_passage_meshes_folder.replace("\\", "/")
        source_output_folder = source_output_folder.replace("\\", "/")
        safe_passage_name = str(passage_name).replace("'", "")

        return rf"""
# This script is written to a real .sh file and then executed by WSL.
# That avoids bash -c quoting/variable-expansion problems from Windows.
# Use an APT-Grid-specific variable name to avoid any OpenFOAM/internal conflicts.
if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc" || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f /opt/openfoam12/etc/bashrc ]; then
        . /opt/openfoam12/etc/bashrc || true
    elif [ -f /usr/lib/openfoam/openfoam12/etc/bashrc ]; then
        . /usr/lib/openfoam/openfoam12/etc/bashrc || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    echo "[Error] blockMesh was not found after loading OpenFOAM."
    exit 127
fi

set -e

AG_RUN_ROOT="/tmp/AptGridOpenFOAM_aptgrid"
SRC_PASSAGE_MESHES={self.shell_quote(source_passage_meshes_folder)}
SRC_OUTPUT={self.shell_quote(source_output_folder)}
PASSAGE_NAME={self.shell_quote(safe_passage_name)}

rm -rf "$AG_RUN_ROOT"
mkdir -p "$AG_RUN_ROOT/passageMeshes" "$AG_RUN_ROOT/outputData"

cp -a "$SRC_PASSAGE_MESHES/." "$AG_RUN_ROOT/passageMeshes/"
cp -a "$SRC_OUTPUT/." "$AG_RUN_ROOT/outputData/"

cd "$AG_RUN_ROOT/passageMeshes"
echo "[OpenFOAM staging folder] $(pwd)"

find . -name "*.sh" -exec sed -i 's/\r$//' {{}} \;
mkdir -p template/constant/geometry template/system

if [ -f template/system/blockMeshDict ]; then
    sed -i 's/#eval{{\([^}}]*\)}}/#calc "\1"/g' template/system/blockMeshDict
fi

if [ ! -d "$AG_RUN_ROOT/outputData/$PASSAGE_NAME" ]; then
    echo "[Error] Missing generated folder: $AG_RUN_ROOT/outputData/$PASSAGE_NAME"
    exit 2
fi

if ! ls "$AG_RUN_ROOT/outputData/$PASSAGE_NAME"/*.stl >/dev/null 2>&1; then
    echo "[Error] No STL files found in: $AG_RUN_ROOT/outputData/$PASSAGE_NAME"
    exit 2
fi

if [ ! -f "$AG_RUN_ROOT/outputData/$PASSAGE_NAME/passageParameters" ]; then
    echo "[Error] Missing passageParameters in: $AG_RUN_ROOT/outputData/$PASSAGE_NAME"
    exit 2
fi

rm -rf "$PASSAGE_NAME"
mkdir -p "$PASSAGE_NAME"
cp -a template/. "$PASSAGE_NAME/"
mkdir -p "$PASSAGE_NAME/constant/geometry" "$PASSAGE_NAME/system"
cp "$AG_RUN_ROOT/outputData/$PASSAGE_NAME"/*.stl "$PASSAGE_NAME/constant/geometry/"
cp "$AG_RUN_ROOT/outputData/$PASSAGE_NAME/passageParameters" "$PASSAGE_NAME/system/"

cd "$PASSAGE_NAME"

if [ ! -f ./geomUpdate.sh ]; then
    echo "[Error] Missing geomUpdate.sh in case folder: $(pwd)"
    exit 2
fi

bash ./geomUpdate.sh

if [ -f system/blockMeshDict ]; then
    sed -i 's/#eval{{\([^}}]*\)}}/#calc "\1"/g' system/blockMeshDict
fi

unset FOAM_CASE
export FOAM_CASE="$(pwd)"
echo "[OpenFOAM case folder] $(pwd)"

blockMesh
CHECKMESH_STATUS=0
checkMesh || CHECKMESH_STATUS=$?

# Save the finished OpenFOAM mesh back to the original folder,
# matching the original workflow output location:
# APT-Grid/passageMeshes/passage0/constant/polyMesh
DEST_CASE="$SRC_PASSAGE_MESHES/$PASSAGE_NAME"
SRC_CASE="$AG_RUN_ROOT/passageMeshes/$PASSAGE_NAME"
DEST_POLYMESH="$DEST_CASE/constant/polyMesh"
echo "[Saving OpenFOAM mesh] $DEST_POLYMESH"

copy_plain_tree() {{
    SRC_DIR="$1"
    DST_DIR="$2"
    if [ ! -d "$SRC_DIR" ]; then
        return 0
    fi
    rm -rf "$DST_DIR"
    mkdir -p "$DST_DIR"
    (cd "$SRC_DIR" && find . -type d -print) | while IFS= read -r d; do
        mkdir -p "$DST_DIR/$d"
    done
    (cd "$SRC_DIR" && find . -type f -print) | while IFS= read -r f; do
        mkdir -p "$DST_DIR/$(dirname "$f")"
        cat "$SRC_DIR/$f" > "$DST_DIR/$f"
    done
}}

copy_plain_tree "$SRC_CASE/constant/polyMesh" "$DEST_CASE/constant/polyMesh"
copy_plain_tree "$SRC_CASE/constant/geometry" "$DEST_CASE/constant/geometry"
copy_plain_tree "$SRC_CASE/system" "$DEST_CASE/system"
echo "[Saved OpenFOAM mesh] $DEST_POLYMESH"

if [ "$CHECKMESH_STATUS" -ne 0 ]; then
    echo "[Warning] checkMesh returned status $CHECKMESH_STATUS. Mesh files were still saved."
fi
exit 0
""".strip()

    def wsl_multipassage_shell_command(self, source_passage_meshes_folder, source_output_folder):
        # stage everything inside Linux first, then run the existing two-passage script
        source_passage_meshes_folder = source_passage_meshes_folder.replace("\\", "/")
        source_output_folder = source_output_folder.replace("\\", "/")

        return rf"""

if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f "$HOME/.bashrc" ]; then
        . "$HOME/.bashrc" || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    if [ -f /opt/openfoam12/etc/bashrc ]; then
        . /opt/openfoam12/etc/bashrc || true
    elif [ -f /usr/lib/openfoam/openfoam12/etc/bashrc ]; then
        . /usr/lib/openfoam/openfoam12/etc/bashrc || true
    fi
fi

if ! command -v blockMesh >/dev/null 2>&1; then
    echo "[Error] blockMesh was not found after loading OpenFOAM."
    exit 127
fi

set -e

AG_RUN_ROOT="/tmp/AptGridOpenFOAM_aptgrid"
SRC_PASSAGE_MESHES={self.shell_quote(source_passage_meshes_folder)}
SRC_OUTPUT={self.shell_quote(source_output_folder)}

rm -rf "$AG_RUN_ROOT"
mkdir -p "$AG_RUN_ROOT/passageMeshes" "$AG_RUN_ROOT/outputData"

cp -a "$SRC_PASSAGE_MESHES/." "$AG_RUN_ROOT/passageMeshes/"
cp -a "$SRC_OUTPUT/." "$AG_RUN_ROOT/outputData/"

cd "$AG_RUN_ROOT/passageMeshes"
echo "[OpenFOAM staging folder] $(pwd)"

find . -name "*.sh" -exec sed -i 's/\r$//' {{}} \;
mkdir -p template/constant/geometry template/system

if [ -f template/system/blockMeshDict ]; then
    sed -i 's/#eval{{\([^}}]*\)}}/#calc "\1"/g' template/system/blockMeshDict
fi

if [ ! -f multipassagetest.sh ]; then
    echo "[Error] Missing multipassagetest.sh in: $(pwd)"
    exit 2
fi

unset FOAM_CASE
MESH_STATUS=0
bash multipassagetest.sh || MESH_STATUS=$?

# Save generated OpenFOAM passage cases back to the original folder
copy_plain_tree() {{
    SRC_DIR="$1"
    DST_DIR="$2"
    if [ ! -d "$SRC_DIR" ]; then
        return 0
    fi
    rm -rf "$DST_DIR"
    mkdir -p "$DST_DIR"
    (cd "$SRC_DIR" && find . -type d -print) | while IFS= read -r d; do
        mkdir -p "$DST_DIR/$d"
    done
    (cd "$SRC_DIR" && find . -type f -print) | while IFS= read -r f; do
        mkdir -p "$DST_DIR/$(dirname "$f")"
        cat "$SRC_DIR/$f" > "$DST_DIR/$f"
    done
}}

for CASE_DIR in passage*; do
    if [ -d "$CASE_DIR" ]; then
        echo "[Saving OpenFOAM case] $SRC_PASSAGE_MESHES/$CASE_DIR"
        SRC_CASE="$AG_RUN_ROOT/passageMeshes/$CASE_DIR"
        DEST_CASE="$SRC_PASSAGE_MESHES/$CASE_DIR"
        mkdir -p "$DEST_CASE/constant"
        # Keep the useful case pieces, but skip OpenFOAM dynamicCode build artifacts.
        copy_plain_tree "$SRC_CASE/constant/polyMesh" "$DEST_CASE/constant/polyMesh"
        copy_plain_tree "$SRC_CASE/constant/geometry" "$DEST_CASE/constant/geometry"
        copy_plain_tree "$SRC_CASE/system" "$DEST_CASE/system"
    fi
done

echo "[Saved OpenFOAM cases] $SRC_PASSAGE_MESHES"
if [ "$MESH_STATUS" -ne 0 ]; then
    echo "[Warning] multipassagetest.sh returned status $MESH_STATUS. Any generated case files were still saved."
fi
exit 0
""".strip()

    def write_shell_script(self, shell_command, script_filename, output_folder):
        """Write a temporary shell script to output_folder and return its local path."""
        os.makedirs(output_folder, exist_ok=True)
        script_path = os.path.join(output_folder, script_filename)

        with open(script_path, "w", encoding="utf-8", newline="\n") as script_file:
            script_file.write("#!/usr/bin/env bash\n")
            script_file.write(shell_command.strip())
            script_file.write("\n")

        try:
            os.chmod(script_path, 0o755)
        except OSError:
            pass

        return script_path

    def write_wsl_shell_script(self, shell_command, script_filename, output_folder):
        """Write a temporary WSL shell script to output_folder and return its WSL path."""
        script_path = self.write_shell_script(shell_command, script_filename, output_folder)
        return self.windows_to_wsl_path(script_path)

    def build_multipassage_command(self, output_folder):
        working_folder = self.get_passage_meshes_folder()
        bash_script = self.find_multipassage_script()

        if bash_script is None:
            return None, working_folder, None

        if os.name == "nt":
            if shutil.which("wsl"):
                wsl_passage_meshes_folder = self.windows_to_wsl_path(working_folder)
                wsl_output_folder = self.windows_to_wsl_path(output_folder)
                shell_command = self.wsl_multipassage_shell_command(wsl_passage_meshes_folder, wsl_output_folder)
                wsl_script_path = self.write_wsl_shell_script(shell_command, "aptgrid_openfoam_multipassage.sh", output_folder)
                return ["wsl", "bash", wsl_script_path], working_folder, bash_script

            bash_exe = shutil.which("bash")
            if bash_exe:
                shell_command = self.wsl_multipassage_shell_command(
                    working_folder.replace("\\", "/"),
                    output_folder.replace("\\", "/")
                )
                script_path = self.write_shell_script(shell_command, "aptgrid_openfoam_multipassage.sh", output_folder)
                return [bash_exe, script_path], working_folder, bash_script

            return None, working_folder, bash_script

        # if native linux/macOS path: run OpenFOAM directly with bash, no WSL path conversion
        shell_command = self.wsl_multipassage_shell_command(
            working_folder.replace("\\", "/"),
            output_folder.replace("\\", "/")
        )
        script_path = self.write_shell_script(shell_command, "aptgrid_openfoam_multipassage.sh", output_folder)
        return ["bash", script_path], working_folder, bash_script

    def build_bash_command(self, bash_script, working_folder):
        if os.name == "nt":
            if shutil.which("wsl"):
                wsl_working_folder = self.windows_to_wsl_path(working_folder)
                wsl_script = self.windows_to_wsl_path(bash_script)
                script_name = os.path.basename(wsl_script)

                if os.path.dirname(bash_script) == working_folder:
                    shell_command = (
                        f'cd "{wsl_working_folder}" || exit 1; '
                        'find . -name "*.sh" -exec sed -i \'s/\\r$//\' {} \\; 2>/dev/null || true; '
                        'mkdir -p template/constant/geometry template/system; '
                        'sed -i \'s/#eval{\\([^}]*\\)}/#calc "\\1"/g\' template/system/blockMeshDict 2>/dev/null || true; '
                        f'bash "{script_name}"'
                    )
                else:
                    shell_command = (
                        f'cd "{wsl_working_folder}" || exit 1; '
                        f'sed -i \'s/\\r$//\' "{wsl_script}" 2>/dev/null || true; '
                        f'bash "{wsl_script}"'
                    )

                return ["wsl", "bash", "-lc", shell_command]

            bash_exe = shutil.which("bash")
            if bash_exe:
                return [bash_exe, bash_script]

            return None

        return ["bash", bash_script]

    def build_single_passage_command(self, output_folder, passage_name="passage0"):
        working_folder = self.get_passage_meshes_folder()

        if os.name == "nt":
            if shutil.which("wsl"):
                wsl_passage_meshes_folder = self.windows_to_wsl_path(working_folder)
                wsl_output_folder = self.windows_to_wsl_path(output_folder)
                shell_command = self.wsl_single_passage_shell_command(
                    wsl_passage_meshes_folder,
                    wsl_output_folder,
                    passage_name
                )
                wsl_script_path = self.write_wsl_shell_script(shell_command, "aptgrid_openfoam_single.sh", output_folder)
                return ["wsl", "bash", wsl_script_path], working_folder

            bash_exe = shutil.which("bash")
            if bash_exe:
                shell_command = self.wsl_single_passage_shell_command(
                    working_folder.replace("\\", "/"),
                    output_folder.replace("\\", "/"),
                    passage_name
                )
                script_path = self.write_shell_script(shell_command, "aptgrid_openfoam_single.sh", output_folder)
                return [bash_exe, script_path], working_folder

            return None, working_folder

        shell_command = self.wsl_single_passage_shell_command(
            working_folder.replace("\\", "/"),
            output_folder.replace("\\", "/"),
            passage_name
        )
        script_path = self.write_shell_script(shell_command, "aptgrid_openfoam_single.sh", output_folder)
        return ["bash", script_path], working_folder
