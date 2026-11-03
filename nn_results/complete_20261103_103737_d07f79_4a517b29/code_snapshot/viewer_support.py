"""Find native OpenSim viewer and geometry."""
import os,sys,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
def prepare_display(osim, source, geometry_folder, live):
    """Find local mesh files and native viewer; never change the source model."""
    package = Path(osim.__file__).resolve().parent
    installs = []
    for base in (Path(os.environ.get("ProgramFiles", "C:/Program Files")),
                 Path("C:/"), Path.home() / "Documents"):
        if base.is_dir() and (os.name == "nt" or base == Path.home() / "Documents"):
            installs.extend(p for p in base.glob("OpenSim*") if p.is_dir())
    if os.environ.get("OPENSIM_HOME"):
        installs.append(Path(os.environ["OPENSIM_HOME"]))
    candidates = [source.parent, source.parent / "Geometry", package / "Geometry"]
    for parent in list(source.parents)[:4]:
        candidates.append(parent / "Geometry")
    for install in installs:
        candidates.extend([install / "Geometry", install / "Resources" / "Geometry",
                           install / "Models" / "Geometry"])
    if geometry_folder:
        candidates.insert(0, Path(geometry_folder).expanduser().resolve())
    mesh_refs = sorted({node.text.strip() for node in ET.parse(source).iter("mesh_file")
                        if node.text and node.text.strip()})
    registered = set()
    def register(folder):
        if folder.is_dir() and folder.resolve() not in registered:
            osim.ModelVisualizer.addDirToGeometrySearchPaths(str(folder.resolve()))
            registered.add(folder.resolve())
    for folder in candidates:
        register(folder)
    def missing():
        return [ref for ref in mesh_refs if not (Path(ref).is_absolute() and Path(ref).is_file())
                and not any((p / ref).is_file() for p in registered)]
    # The GUI's geometry preferences are not inherited by Python. Search only
    # model/project and OpenSim installation trees, not the entire computer.
    if live and missing():
        trees = [source.parent] + installs
        if geometry_folder:
            trees.insert(0, Path(geometry_folder).expanduser().resolve())
        for parent in list(source.parents)[1:4]:
            if parent.name.lower() in ("models", "opensim"):
                trees.append(parent)
        seen = set()
        for tree in trees:
            if not tree.is_dir() or tree.resolve() in seen:
                continue
            seen.add(tree.resolve())
            wanted = {Path(ref).name for ref in missing()}
            for mesh in tree.rglob("*.vtp"):
                if mesh.name in wanted:
                    register(mesh.parent)
            if not missing():
                break
    if live and missing():
        print("Python needs the arm26 Geometry files to display the bones.\n"
              "Select your OpenSim Geometry folder in the folder-selection dialog.", flush=True)
        try:
            import tkinter as tk
            from tkinter import filedialog
            window = tk.Tk()
            window.withdraw()
            try:
                selected = filedialog.askdirectory(title="Select Geometry folder containing ground_ribs.vtp",
                                                    initialdir=str(source.parent), parent=window)
            finally:
                window.destroy()
            if selected:
                register(Path(selected))
                wanted = {Path(ref).name for ref in missing()}
                for mesh in Path(selected).rglob("*.vtp"):
                    if mesh.name in wanted:
                        register(mesh.parent)
        except Exception as exc:
            print(f"Folder chooser unavailable: {exc}")
        if missing():
            raise FileNotFoundError("Cannot display all bones. Missing examples: " +
                                    ", ".join(missing()[:4]) +
                                    '\nRerun with --geometry "FULL_PATH_TO_YOUR_GEOMETRY_FOLDER".')
    if not live:
        return
    binary = "simbody-visualizer.exe" if os.name == "nt" else "simbody-visualizer"
    binary_dirs = [package, package / "bin", package.parent / "opensim.libs",
                   Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "bin"]
    binary_dirs += [install / "bin" for install in installs]
    found = next((folder / binary for folder in binary_dirs if (folder / binary).is_file()), None)
    if found is None and shutil.which(binary):
        found = Path(shutil.which(binary))
    if found is None:
        raise FileNotFoundError("The OpenSim simulation viewer (" + binary + ") was not found. "
                               "Send this message and the printed Python interpreter path. "
                               "--no-live can still save plots, but will not show the 3D model.")
    search_dirs = [found.parent] + [p for p in binary_dirs if p.is_dir()]
    os.environ["PATH"] = os.pathsep.join(map(str, search_dirs)) + os.pathsep + os.environ.get("PATH", "")
    print(f"3D viewer: {found}", flush=True)
