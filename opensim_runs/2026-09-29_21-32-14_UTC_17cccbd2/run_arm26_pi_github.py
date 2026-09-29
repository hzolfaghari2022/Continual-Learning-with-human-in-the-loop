"""Sampled feedback control of the arm26 flexor cable (OpenSim 4.6).

Run from VS Code with your working OpenSim Python interpreter.
Input and results are resolved beside THIS script, independently of terminal cwd.
This is a controller-development example, NOT the paper's reproduced experiment.
This is the FULL runner, not the one-time updater.
The source .osim is never modified. Opens live 3D and tracking plots automatically.
After completion, Replay replays the recorded simulated states. No manual loading.
"""
import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import subprocess
import threading
import json
from datetime import datetime, timezone
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET
import hashlib
import tempfile
import uuid
import re
import stat

INPUT_MODEL = "arm26_exosuit_with_cable_2.osim"
# Edit these values or use command-line options. All internal angles are radians.
KP = 30.0        # proportional gain, N m / rad
KI = 15.0        # integral gain, N m / (rad s)
KV = 0.0         # PURE PI by default; try 3.0 for optional velocity damping
DT = 0.01       # controller period, seconds (100 Hz)
DEMO_START_DEG = 20.0
DEMO_TARGET_DEG = 80.0
DEMO_CYCLES = 2
DURATION = 13.5  # 0.5 s start + four 3 s legs + 1 s final hold
RISE_TIME = 3.0  # seconds per bending or returning leg
TENSION_MAX = 80.0  # N; example actuator limit, NOT a limit from the paper
EXCITATION = 0.01   # fixed low excitation for each of the six muscles

# Automatic upload uses your existing GitHub credentials on THIS computer.
AUTO_PUSH_TO_GITHUB = True
GITHUB_REPO_URL = "https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git"
GITHUB_BRANCH = "main"
GITHUB_RESULTS_FOLDER = "opensim_runs"
GITHUB_COMMIT_NAME = "hzolfaghari2022"
GITHUB_COMMIT_EMAIL = "105059650+hzolfaghari2022@users.noreply.github.com"


def github_identity(url):
    """Compare HTTPS and SSH GitHub addresses without printing credentials."""
    if url.startswith("git@github.com:"):
        path = url[len("git@github.com:"):]
    else:
        parsed = urlsplit(url)
        if parsed.scheme not in ("https", "ssh") or parsed.hostname != "github.com":
            raise ValueError("Use an HTTPS or SSH repository address on github.com.")
        if parsed.password or (parsed.scheme == "https" and parsed.username) or parsed.query or parsed.fragment:
            raise ValueError("Use a clean repository URL without a token or password.")
        path = parsed.path.lstrip("/")
    path = path.rstrip("/")
    if path.endswith(".git"):
        path = path[:-4]
    parts = path.split("/")
    if len(parts) != 2 or any(not p or p in (".", "..") for p in parts):
        raise ValueError("Repository address must identify exactly one GitHub owner and repository.")
    return path.lower()


GITHUB_FILES = (
    "PI_data.csv", "no_integral_data.csv", "PI_motion.mot", "no_integral_motion.mot",
    "PI_states.sto", "no_integral_states.sto", "arm26_control_playback.osim",
    "control_plots.png", "control_plots.pdf", "controller_diagram.png", "README.txt",
    "input_model.osim", "run_arm26_pi_github.py", "run_settings.json",
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_run_snapshot(source, results, args, summaries):
    """Save everything needed for a later upload before opening blocking plot windows."""
    shutil.copy2(source, results / "input_model.osim")
    shutil.copy2(Path(__file__), results / "run_arm26_pi_github.py")
    run_id = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC_") + uuid.uuid4().hex[:8]
    settings = dict(run_id=run_id, completed=True, kp=args.kp, ki=args.ki, kv=args.kv,
                    dt_s=args.dt, duration_s=args.duration, leg_duration_s=args.rise_time,
                    reference_min_deg=DEMO_START_DEG, reference_max_deg=DEMO_TARGET_DEG,
                    cycles=DEMO_CYCLES, max_tension_N=args.tmax, muscle_excitation=EXCITATION,
                    source_model=source.name, summaries=summaries, python_version=sys.version,
                    opensim_version=args.opensim_version)
    (results / "run_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    manifest = dict(run_id=run_id, files={name:file_hash(results/name) for name in GITHUB_FILES})
    (results / "upload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def git_command(git, *command, cwd=None):
    # Git Credential Manager may show its browser sign-in. No tokens are stored here.
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    try:
        proc = subprocess.run([git, *command], cwd=cwd, env=env, text=True,
                              encoding="utf-8", errors="replace", capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Git timed out. Complete GitHub sign-in, check your connection, then retry the upload.") from exc
    if proc.returncode:
        error = proc.stderr.lower()
        if any(word in error for word in ("authentication", "permission denied", "could not read username", "403", "repository not found")):
            hint = ("GitHub authentication/access failed. In a terminal run 'gh auth login' and "
                    "'gh auth setup-git' if GitHub CLI is installed, or sign in with Git Credential Manager.")
        elif any(word in error for word in ("non-fast-forward", "fetch first", "rejected")):
            hint = "The remote changed or rejected the push. Retry; if it persists, check branch protection."
        elif any(word in error for word in ("resolve host", "connect", "ssl", "network")):
            hint = "Git could not reach GitHub. Check your network/proxy and retry."
        else:
            hint = "Check Git installation, GitHub credentials, and repository/branch access."
        # Keep credential-helper output and remote URLs out of published diagnostics.
        raise RuntimeError(f"Git {command[0]} failed (exit {proc.returncode}). {hint}")
    return proc.stdout.strip()


def push_completed_run(results):
    """Use a fresh temporary checkout, leaving existing repositories untouched."""
    messages = []
    temporary = None
    def log(message):
        messages.append(message)
        (results / "github_push_status.txt").write_text("\n".join(messages)+"\n", encoding="utf-8")
        print("GITHUB: " + message, flush=True)
    try:
        identity = github_identity(GITHUB_REPO_URL)
        git = shutil.which("git")
        if not git:
            raise RuntimeError("Git was not found. Install Git for Windows, restart VS Code, then retry the upload.")
        manifest = json.loads((results / "upload_manifest.json").read_text(encoding="utf-8"))
        run_id = manifest["run_id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid run identifier.")
        if set(manifest["files"]) != set(GITHUB_FILES):
            raise ValueError("The snapshot is incomplete or its upload file list was changed.")
        for name, expected in manifest["files"].items():
            path = results / name
            if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"Snapshot file missing or changed: {name}. Keep the completed run intact.")
        folder = Path(GITHUB_RESULTS_FOLDER)
        if folder.is_absolute() or ".." in folder.parts or ".git" in folder.parts or not folder.parts:
            raise ValueError("Invalid GitHub results folder.")
        log(f"Uploading {run_id} to {identity} (branch {GITHUB_BRANCH})...")
        log("A GitHub sign-in window may appear on the first push from this computer.")
        heads = git_command(git, "ls-remote", "--heads", GITHUB_REPO_URL)
        refs = {line.split()[1] for line in heads.splitlines() if line.strip()}
        wanted = "refs/heads/" + GITHUB_BRANCH
        if refs and wanted not in refs:
            raise RuntimeError(f"Branch {GITHUB_BRANCH} is absent. Set GITHUB_BRANCH to an existing branch.")
        temporary = Path(tempfile.mkdtemp(prefix="arm26_github_"))
        repo = temporary / "repo"
        clone_args = ["clone", "--depth", "1", "--single-branch"]
        if refs:
            clone_args.extend(["--branch", GITHUB_BRANCH])
        git_command(git, *clone_args, "--", GITHUB_REPO_URL, str(repo))
        if not refs:
            git_command(git, "symbolic-ref", "HEAD", wanted, cwd=repo)
        git_command(git, "config", "user.name", GITHUB_COMMIT_NAME, cwd=repo)
        git_command(git, "config", "user.email", GITHUB_COMMIT_EMAIL, cwd=repo)
        # Local settings apply only to this disposable clone, never the user's global Git settings.
        git_command(git, "config", "core.autocrlf", "false", cwd=repo)
        git_command(git, "config", "commit.gpgsign", "false", cwd=repo)
        git_command(git, "config", "core.hooksPath", str(temporary / "no-hooks"), cwd=repo)
        relative = (folder / run_id).as_posix()
        snapshot = repo / folder / run_id
        if snapshot.is_symlink() or not snapshot.resolve().is_relative_to(repo.resolve()):
            raise ValueError("Remote results directory resolves outside the checkout.")
        snapshot.mkdir(parents=True, exist_ok=True)
        for name in (*GITHUB_FILES, "upload_manifest.json"):
            target = snapshot / name
            if target.is_symlink():
                raise ValueError("Unexpected symlink in the remote run directory.")
            shutil.copyfile(results / name, target)
        git_command(git, "--literal-pathspecs", "add", "--", relative, cwd=repo)
        changed = git_command(git, "diff", "--cached", "--name-only", cwd=repo).splitlines()
        if not changed:
            commit = git_command(git, "rev-parse", "HEAD", cwd=repo)
            log(f"SUCCESS: already uploaded. https://github.com/{identity}/tree/{commit}/{relative}")
            return True
        if any(not p.startswith(relative + "/") for p in changed):
            raise RuntimeError("Unexpected staged files; upload stopped.")
        tracked = set(git_command(git, "--literal-pathspecs", "ls-files", "--", relative, cwd=repo).splitlines())
        if tracked != {relative+"/"+name for name in (*GITHUB_FILES, "upload_manifest.json")}:
            raise RuntimeError("Some snapshot files were ignored or unexpected files exist in the run folder.")
        git_command(git, "commit", "-m", f"Save OpenSim PI trajectory run {run_id}", cwd=repo)
        commit = git_command(git, "rev-parse", "HEAD", cwd=repo)
        git_command(git, "push", "origin", f"HEAD:{wanted}", cwd=repo)
        log(f"SUCCESS: https://github.com/{identity}/tree/{commit}/{relative}")
        return True
    except Exception as exc:
        log(f"NOT PUSHED: {exc}")
        log("Simulation results remain saved. Retry WITHOUT rerunning the simulation:")
        command = subprocess.list2cmdline([sys.executable, str(Path(__file__).resolve()), "--push-results", str(results)])
        log(("& " if os.name == "nt" else "") + command)
        return False
    finally:
        if temporary:
            def writable_then_remove(function, path, exc_info):
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                function(path)
            try:
                shutil.rmtree(temporary, onerror=writable_then_remove)
            except OSError:
                print(f"Temporary Git checkout retained at {temporary}", flush=True)


def start_github_push(results, args):
    if not AUTO_PUSH_TO_GITHUB or args.no_push:
        (results / "github_push_status.txt").write_text("Automatic push disabled for this run.\n", encoding="utf-8")
        return None
    worker = threading.Thread(target=push_completed_run, args=(results,), daemon=False)
    worker.start()
    return worker


def viewer_error(args, stage, exc):
    message = f"VIEWER PROBLEM at {stage}: {type(exc).__name__}: {exc}\n"
    print(message, flush=True)
    if hasattr(args, "results_dir"):
        path = args.results_dir / "viewer_error.txt"
        with path.open("a", encoding="utf-8") as f:
            f.write(message + traceback.format_exc() + "\n")
        print(f"Details saved to: {path}", flush=True)


def find_input(root, requested):
    if requested:
        candidates = [(root / requested).resolve()]
    else:
        candidates = [p / INPUT_MODEL for p in (root, root.parent, root.parent.parent)]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("Cable model not found. Checked:\n" +
                            "\n".join(str(p) for p in candidates) +
                            "\nUse --input with your existing cable model's full path.")


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


class LiveDisplay:
    """Show actual simulated states, with synchronized live plots and replay."""
    def __init__(self, osim, model, args, plt, state):
        from matplotlib.widgets import Button
        self.osim, self.args, self.plt = osim, args, plt
        self.simulation_model = model
        # A separate display model prevents a broken native window from killing integration.
        self.model = osim.Model(model)
        self.model.setUseVisualizer(True)
        self.viewer_state = self.model.initSystem()
        original_names = model.getStateVariableNames()
        display_names = self.model.getStateVariableNames()
        if ([original_names.get(i) for i in range(original_names.getSize())] !=
                [display_names.get(i) for i in range(display_names.getSize())]):
            raise RuntimeError("Display/physics state layouts differ; visualization disabled.")
        self.viewer_failed = False
        self.copy_display_state(state)
        self.viewer = self.model.updVisualizer()
        self.native = self.viewer.updSimbodyVisualizer()
        # Draw first. Optional appearance/enum bindings must not prevent frames.
        print("VIEWER: drawing the initial model state...", flush=True)
        self.viewer.show(self.viewer_state)
        self.configure_view()
        self.viewer.show(self.viewer_state)
        print("VIEWER: initial frame sent. Opening all six live plots...", flush=True)
        self.frames, self.rows = [], []
        self.finished = False
        self.next_frame = 0.0
        self.replaying, self.replay_index = False, 0
        self.fig, grid = plt.subplots(3, 2, figsize=(12, 9))
        self.axes = list(grid.flat)
        self.fig.subplots_adjust(top=.85, bottom=.16, hspace=.48, wspace=.32, left=.08, right=.97)
        self.title = self.fig.suptitle("LIVE PI control — actual OpenSim simulation", fontsize=14)
        times = [k * args.duration / 400 for k in range(401)]
        self.axes[0].plot(times, [math.degrees(reference(t, args.rise_time)) for t in times],
                          "k--", label="Desired", linewidth=2)
        self.angle_line, = self.axes[0].plot([], [], color="#008aab", label="Actual", linewidth=2)
        self.error_line, = self.axes[1].plot([], [], color="#ad661e")
        self.force_line, = self.axes[2].plot([], [], color="#008e64")
        self.extra_lines = []
        for key,label in [("P_Nm","P"),("I_Nm","I"),("torque_applied_Nm","Applied total")]:
            line, = self.axes[3].plot([],[],label=label)
            self.extra_lines.append((line,key))
        for index, suffix in [(4,"_fiber_length_m"),(5,"_tendon_force_N")]:
            for muscle,label in [("BIClong","Biceps long"),("TRIlong","Triceps long")]:
                line, = self.axes[index].plot([],[],label=label)
                self.extra_lines.append((line,muscle+suffix))
        self.axes[0].set(ylabel="Elbow angle (deg)", ylim=(DEMO_START_DEG-5, DEMO_TARGET_DEG+5))
        self.axes[0].legend(loc="upper left")
        self.axes[1].set(ylabel="Error (deg)", ylim=(-8, 8))
        self.axes[1].axhline(0, color="gray", linewidth=.6)
        self.axes[2].set(ylabel="Cable tension (N)", xlabel="Time (s)", ylim=(0,args.tmax*1.1))
        self.axes[3].set(ylabel="Controller torque (N m)")
        self.axes[4].set(ylabel="Muscle fiber length (m)")
        self.axes[5].set(ylabel="Muscle tendon force (N)")
        for index in (3,4,5):
            self.axes[index].legend(fontsize=8)
        self.cursors = []
        for ax in self.axes:
            ax.set_xlim(0, args.duration)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=.2)
            self.cursors.append(ax.axvline(0,color="#666666",linewidth=1))
        self.status = self.fig.text(.13,.93,"",fontsize=10)
        self.note = self.fig.text(.13,.105,"Gravity on | Fixed muscle excitation: 0.01 | Live simulation",fontsize=9)
        self.button = Button(self.fig.add_axes([.36,.025,.3,.05]), "Replay after completion")
        self.button.on_clicked(self.start_replay)
        from matplotlib.widgets import Button as ResetButton
        self.reset_button = ResetButton(self.fig.add_axes([.70,.025,.22,.05]),"Reset 3D camera")
        self.reset_button.on_clicked(lambda event:self.configure_view())
        self.timer = self.fig.canvas.new_timer(interval=33)
        self.timer.add_callback(self.replay_tick)
        plt.show(block=False)
        self.fig.canvas.draw()
        plt.pause(.1)
        self.wall_start = time.perf_counter()

    def copy_display_state(self, state):
        self.viewer_state.setTime(state.getTime())
        values = self.simulation_model.getStateVariableValues(state)
        self.model.setStateVariableValues(self.viewer_state, values)
        self.model.realizeVelocity(self.viewer_state)

    def configure_view(self):
        # Integer helper is supplied by OpenSim to avoid nested enum issues.
        actions = [
            ("background",lambda:self.native.setBackgroundTypeByInt(2)),
            ("color",lambda:self.native.setBackgroundColor(self.osim.Vec3(.75,.78,.82))),
            ("clipping",lambda:self.native.setCameraClippingPlanes(.001,100)),
            ("fit camera",lambda:self.native.zoomCameraToShowAllGeometry()),
        ]
        for stage,action in actions:
            try:
                action()
            except Exception as exc:
                viewer_error(self.args,stage,exc)

    def show_frame(self, state, row, mode):
        if not self.viewer_failed:
            try:
                self.copy_display_state(state)
                self.viewer.show(self.viewer_state)
            except Exception as exc:
                self.viewer_failed = True
                viewer_error(self.args, "3D display (physics continues)", exc)
                print("3D window unavailable; simulation, plots and GitHub upload continue.", flush=True)
        for cursor in self.cursors:
            cursor.set_xdata([row["time_s"],row["time_s"]])
        self.status.set_text(f"{mode} | t={row['time_s']:.2f}s | desired={row['reference_deg']:.1f}° | "
                             f"actual={row['angle_deg']:.1f}° | tension={row['tension_N']:.1f} N")
        self.fig.canvas.draw_idle()

    def update(self, state, row, final=False):
        self.rows.append(row)
        if row["time_s"] + 1e-9 < self.next_frame and not final:
            return
        self.next_frame = row["time_s"] + 1/30
        frame_state = self.osim.State(state)
        self.frames.append((frame_state,row))
        times = [r["time_s"] for r in self.rows]
        for line, key in [(self.angle_line,"angle_deg"),(self.error_line,"error_deg"),
                          (self.force_line,"tension_N")] + self.extra_lines:
            line.set_data(times,[r[key] for r in self.rows])
        for ax in self.axes[3:]:
            ax.relim()
            ax.autoscale_view(scalex=False,scaley=True)
        angles = [r["angle_deg"] for r in self.rows]
        self.axes[0].set_ylim(min(20,min(angles)-3),max(DEMO_TARGET_DEG+5,max(angles)+3))
        max_error = max(8, max(abs(r["error_deg"]) for r in self.rows)*1.1)
        self.axes[1].set_ylim(-max_error,max_error)
        self.show_frame(state,row,"LIVE")
        # Pace visualization only; simulated controller period remains args.dt.
        delay = max(.001, row["time_s"] - (time.perf_counter()-self.wall_start))
        self.plt.pause(min(delay,.05))

    def finish(self):
        self.finished = True
        self.title.set_text("PI simulation complete — press Replay to present the result")
        self.button.label.set_text("Replay / Pause")
        self.note.set_text("Replay shows saved simulated states; rerun Python to test new gains.")
        self.fig.canvas.draw_idle()

    def start_replay(self, event=None):
        if not self.finished:
            return
        if self.replaying:
            self.timer.stop()
            self.replaying=False
            return
        if self.replay_index >= len(self.frames):
            self.replay_index=0
        self.replay_start=time.perf_counter()-self.frames[self.replay_index][1]["time_s"]
        self.replaying=True
        self.timer.start()

    def replay_tick(self):
        if not self.replaying:
            return
        elapsed=time.perf_counter()-self.replay_start
        while self.replay_index+1 < len(self.frames) and self.frames[self.replay_index+1][1]["time_s"]<=elapsed:
            self.replay_index+=1
        state,row=self.frames[self.replay_index]
        try:
            self.show_frame(state,row,"REPLAY")
        except Exception as exc:
            self.timer.stop()
            self.replaying=False
            print(f"Replay viewer closed or unavailable: {exc}")
            return
        if self.replay_index==len(self.frames)-1:
            self.timer.stop()
            self.replaying=False
            self.replay_index=len(self.frames)

    def close(self):
        self.timer.stop()
        try:
            self.native.shutdown()
        except Exception:
            pass


def reference(t, rise_time):
    """Two smooth bend-and-return cycles; angles returned in radians."""
    elapsed = max(0.0, t - 0.5)
    legs = 2 * DEMO_CYCLES
    if elapsed >= legs * rise_time:
        return math.radians(DEMO_START_DEG)
    leg = int(elapsed / rise_time)
    u = min(1.0, max(0.0, (elapsed - leg * rise_time) / rise_time))
    smooth = 10*u**3 - 15*u**4 + 6*u**5
    fraction = smooth if leg % 2 == 0 else 1.0 - smooth
    return math.radians(DEMO_START_DEG + (DEMO_TARGET_DEG - DEMO_START_DEG) * fraction)



def write_table(path, names, values, degrees=False):
    with path.open("w", encoding="utf-8") as f:
        f.write(f"{path.stem}\nversion=1\nnRows={len(values)}\nnColumns={len(names)}\n")
        f.write(f"inDegrees={'yes' if degrees else 'no'}\nendheader\n")
        f.write("\t".join(names) + "\n")
        for row in values:
            f.write("\t".join(f"{v:.10g}" for v in row) + "\n")


def run_trial(osim, source, dest, args, ki, name):
    model = osim.Model(str(source))
    if model.getControllerSet().getSize():
        raise ValueError("Use the cable model BEFORE adding any controllers, not a diagnostic model.")
    if not model.getForceSet().contains("exo_flexor"):
        raise ValueError("The input must contain the PathActuator named exo_flexor.")
    cable = osim.PathActuator.safeDownCast(model.updForceSet().get("exo_flexor"))
    if cable is None:
        raise ValueError("exo_flexor is not a PathActuator.")
    muscles = model.updMuscles()
    if muscles.getSize() != 6 or model.getForceSet().getSize() != 7:
        raise ValueError("This example expects the six arm26 muscles and one cable only.")
    model.setName("arm26_control_playback")
    model.setDescription("Playback model. The sampled feedback controller runs in Python, not in this XML. "
                         "Gravity on; six muscles have fixed low excitation. Load the matching .mot.")
    model.setGravity(osim.Vec3(0, -9.80665, 0))
    for i in range(model.getForceSet().getSize()):
        model.updForceSet().get(i).set_appliesForce(True)
    shoulder = model.updCoordinateSet().get("r_shoulder_elev")
    elbow = model.updCoordinateSet().get("r_elbow_flex")
    shoulder.setDefaultValue(0.0)
    shoulder.setDefaultSpeedValue(0.0)
    shoulder.setDefaultLocked(True)
    elbow.setDefaultValue(math.radians(DEMO_START_DEG))
    elbow.setDefaultSpeedValue(0.0)
    elbow.setDefaultLocked(False)
    controller = osim.PrescribedController()
    controller.setName("fixed_muscle_excitation_NOT_the_feedback_controller")
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        thelen = osim.Thelen2003Muscle.safeDownCast(muscle)
        if thelen is None:
            raise ValueError("This example expects the original Thelen2003 muscles.")
        thelen.setDefaultActivation(EXCITATION)
        controller.addActuator(muscle)
        controller.prescribeControlForActuator(muscle.getName(), osim.Constant(EXCITATION))
    model.addController(controller)
    model.finalizeConnections()
    live = args.live and name == "PI"
    # The force-driven simulation model NEVER owns a visualizer/event reporter.
    model.setUseVisualizer(False)
    state = model.initSystem()
    state.setTime(0)
    model.equilibrateMuscles(state)
    # Direct ideal tension input in N. The override bypasses optimal_force scaling.
    # OpenSim still applies that tension along the GeometryPath to both bodies.
    cable.overrideActuation(state, True)
    display = None
    if live:
        try:
            display = LiveDisplay(osim, model, args, args.plt, state)
        except Exception as exc:
            viewer_error(args,"first frame / live dashboard",exc)
            print("Continuing to compute and save the six final plots.",flush=True)
    if display:
        args.display = display  # Keep model and native viewer alive during Replay.
    integral = 0.0
    rows, state_rows = [], []
    state_names = model.getStateVariableNames()
    muscle_names = [muscles.get(i).getName() for i in range(muscles.getSize())]
    print(f"Running {name}: Kp={args.kp}, Ki={ki}, Kv={args.kv}", flush=True)
    steps = round(args.duration / args.dt)
    for k in range(steps + 1):
        t = float(state.getTime())
        if k and k % max(1, round(2.0 / args.dt)) == 0:
            print(f"  {name}: {t:.1f} s simulated", flush=True)
        model.realizeVelocity(state)
        q = elbow.getValue(state)
        speed = elbow.getSpeedValue(state)
        qref = reference(t, args.rise_time)
        error = qref - q
        moment_arm = cable.computeMomentArm(state, elbow)
        if not 0 <= math.degrees(q) <= 90 or moment_arm <= 0.005:
            raise RuntimeError(f"{name} left the verified geometry range at t={t:.3f}s, "
                               f"angle={math.degrees(q):.2f} deg. Adjust gains; no success claimed.")
        p = args.kp * error
        damping = -args.kv * speed
        i_term = ki * integral
        tau_raw = p + i_term + damping
        tension = min(args.tmax, max(0.0, tau_raw / moment_arm))
        tau_applied = tension * moment_arm
        saturated = abs(tau_raw - tau_applied) > 1e-9
        # Conditional integration: freeze only if error would worsen saturation.
        can_integrate = not ((tau_raw >= args.tmax * moment_arm and error > 0)
                             or (tau_raw <= 0 and error < 0))
        cable.setOverrideActuation(state, tension)
        model.realizeDynamics(state)
        measured_tension = cable.getActuation(state)
        if abs(measured_tension - tension) > 1e-6:
            raise RuntimeError("Applied cable tension did not match the command.")
        row = dict(time_s=t, reference_deg=math.degrees(qref), angle_deg=math.degrees(q),
                   error_deg=math.degrees(error), velocity_rad_s=speed,
                   tension_N=measured_tension, moment_arm_m=moment_arm,
                   torque_raw_Nm=tau_raw, torque_applied_Nm=tau_applied,
                   P_Nm=p, I_Nm=i_term, damping_Nm=damping,
                   saturated=int(saturated), fixed_excitation=EXCITATION)
        for i, muscle_name in enumerate(muscle_names):
            muscle = muscles.get(i)
            row[muscle_name + "_activation"] = muscle.getActivation(state)
            row[muscle_name + "_fiber_length_m"] = muscle.getFiberLength(state)
            row[muscle_name + "_path_length_m"] = muscle.getLength(state)
            row[muscle_name + "_tendon_force_N"] = muscle.getTendonForce(state)
        if not all(math.isfinite(v) for v in row.values()):
            raise RuntimeError("Nonfinite result; simulation stopped.")
        rows.append(row)
        values = model.getStateVariableValues(state)
        state_rows.append([t] + [values.get(j) for j in range(values.size())])
        if display:
            try:
                display.update(state, row, final=k == steps)
            except Exception as exc:
                viewer_error(args,"live update",exc)
                display.close()
                display = None
        if k == steps:
            break
        if can_integrate:
            integral += error * args.dt
        # New Manager after the input discontinuity avoids reuse of stale stages.
        # Copy the returned State BEFORE destroying its Manager at the next step.
        manager = osim.Manager(model)
        manager.setIntegratorAccuracy(1e-6)
        manager.setIntegratorMaximumStepSize(args.dt / 5)
        manager.initialize(state)
        state = osim.State(manager.integrate((k + 1) * args.dt))
    with (dest / f"{name}_data.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_table(dest / f"{name}_motion.mot", ["time", "r_shoulder_elev", "r_elbow_flex"],
                [[r["time_s"], 0, r["angle_deg"]] for r in rows], True)
    write_table(dest / f"{name}_states.sto", ["time"] + [state_names.get(j) for j in range(state_names.getSize())], state_rows)
    if name == "PI":
        if not model.printToXML(str(dest / "arm26_control_playback.osim")):
            raise RuntimeError("Could not save playback model.")
    rmse = math.sqrt(sum(r["error_deg"]**2 for r in rows) / len(rows))
    result = f"{name}: RMSE {rmse:.3f} deg; final error {rows[-1]['error_deg']:.3f} deg; "
    result += f"peak tension {max(r['tension_N'] for r in rows):.2f} N"
    print(result, flush=True)
    if display:
        display.finish()
    return rows, result


def plot_results(plt, baseline, controlled, folder, args):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(3, 2, figsize=(13, 11), layout="constrained")
    t = [r["time_s"] for r in controlled]
    def series(rows, key):
        return [r[key] for r in rows]
    def line(a, rows, key, label, **kw):
        a.plot(t, series(rows, key), label=label, **kw)
    suffix = " + velocity damping" if args.kv else " (no velocity feedback)"
    fig.suptitle("Elbow cable control: PI" + suffix + "\nOpenSim simulation | gravity on | muscle excitation fixed at 0.01", fontsize=16)
    a = ax[0, 0]
    line(a, controlled, "reference_deg", "Desired", color="black", linestyle="--", linewidth=2)
    line(a, baseline, "angle_deg", "Without integral", color="#b58c65")
    line(a, controlled, "angle_deg", "With integral", color="#008aab", linewidth=2)
    a.set(title="1. Does the elbow follow the target?", ylabel="Elbow angle (degrees)")
    a = ax[0, 1]
    line(a, baseline, "error_deg", "Without integral", color="#b58c65")
    line(a, controlled, "error_deg", "With integral", color="#008aab")
    a.axhline(0, color="gray", linewidth=0.7)
    a.set(title="2. Does integral action remove the offset?", ylabel="Desired - actual (degrees)")
    a = ax[1, 0]
    line(a, controlled, "tension_N", "Cable tension", color="#009a6c")
    a.axhline(args.tmax, color="gray", linestyle=":", label="Example tension limit")
    a.set(title="3. What does the controller command?", ylabel="Tension (N)", ylim=(-2, args.tmax * 1.1))
    a = ax[1, 1]
    terms = [("P_Nm", "P"), ("I_Nm", "I")]
    if args.kv:
        terms.append(("damping_Nm", "Velocity damping"))
    terms.append(("torque_applied_Nm", "Applied total"))
    for key, label in terms:
        line(a, controlled, key, label)
    a.set(title="4. Where does the assistive torque come from?", ylabel="Elbow moment (N m)")
    a = ax[2, 0]
    for muscle, label in [("BIClong", "Biceps long"), ("TRIlong", "Triceps long")]:
        line(a, controlled, muscle + "_fiber_length_m", label)
    a.set(title="5. How do muscle fibers change length?", ylabel="Fiber length (m)")
    a = ax[2, 1]
    for muscle, label in [("BIClong", "Biceps long"), ("TRIlong", "Triceps long")]:
        line(a, controlled, muscle + "_tendon_force_N", label)
    a.set(title="6. What forces do these muscles transmit?", ylabel="Tendon force (N)")
    for a in ax.flat:
        a.set_xlabel("Time (s)")
        a.grid(alpha=0.2)
        a.legend(fontsize=8)
    fig.savefig(folder / "control_plots.png", dpi=170)
    fig.savefig(folder / "control_plots.pdf")
    return fig


def control_diagram(plt, folder, args):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")
    def box(x, y, w, h, title, detail, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="none"))
        ax.text(x+w/2,y+h*.68,title,ha="center",va="center",weight="bold",fontsize=13)
        ax.text(x+w/2,y+h*.29,detail,ha="center",va="center",fontsize=10)
    box(.3,3.4,2.0,1.2,"Target motion","Desired elbow angle", "#edf1f4")
    box(3,3.4,3.0,1.2,"Python controller", "Error → PI + speed damping" if args.kv else "Angle error → P + I", "#d7eff5")
    box(7,3.4,4.3,1.2,"OpenSim model","Cable + arm + six muscles + gravity", "#dfefdf")
    def arrow(start, end):
        ax.annotate("",xy=end,xytext=start,arrowprops=dict(arrowstyle="->",lw=2,color="#364b5a"))
    arrow((2.4,4),(2.9,4)); arrow((6.1,4),(6.9,4))
    ax.text(6.5,4.4,"Tension\n(N)",ha="center",fontsize=10)
    ax.plot([9.1,9.1,4.5],[3.25,2.2,2.2],color="#364b5a",lw=2)
    arrow((4.5,2.2),(4.5,3.3))
    ax.text(6.9,1.9,"Measured elbow angle + angular velocity" if args.kv else "Measured elbow angle",ha="center",fontsize=11)
    ax.text(6,5.35,"Your controller closes the loop through Python",ha="center",fontsize=18,weight="bold")
    ax.text(6,.95,f"Every {args.dt:g} s: read the state → calculate tension → advance OpenSim",ha="center",fontsize=12)
    ax.text(6,.4,"Muscle excitation stays at 0.01 in this example; OpenSim calculates muscle dynamics.",ha="center",fontsize=10,color="#52616b")
    fig.savefig(folder / "controller_diagram.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="Defaults to cable model beside script or up to two folders above.")
    parser.add_argument("--geometry", default=None, help="Folder containing arm26 .vtp mesh files.")
    parser.add_argument("--no-live", action="store_true", help="Disable the automatic 3D viewer and live plots.")
    parser.add_argument("--kp", type=float, default=KP)
    parser.add_argument("--ki", type=float, default=KI)
    parser.add_argument("--kv", type=float, default=KV)
    parser.add_argument("--dt", type=float, default=DT)
    parser.add_argument("--duration", type=float, default=None, help="Default: enough time for two full cycles and a final hold.")
    parser.add_argument("--rise-time", type=float, default=RISE_TIME)
    parser.add_argument("--tmax", type=float, default=TENSION_MAX)
    parser.add_argument("--no-show", action="store_true", help="Headless mode: no 3D viewer or plot windows.")
    parser.add_argument("--no-push", action="store_true", help="Save results locally without GitHub upload.")
    parser.add_argument("--push-results", default=None, help="Upload an existing completed result folder without simulating.")
    args = parser.parse_args()
    if args.push_results:
        folder = Path(args.push_results).expanduser()
        if not folder.is_absolute():
            folder = Path(__file__).resolve().parent / folder
        if not folder.is_dir():
            raise FileNotFoundError(f"Results folder not found: {folder}")
        if not push_completed_run(folder.resolve()):
            raise SystemExit(2)
        return
    if args.duration is None:
        args.duration = 0.5 + 2 * DEMO_CYCLES * args.rise_time + 1.0
    if not all(math.isfinite(x) for x in (args.kp,args.ki,args.kv,args.dt,args.duration,args.rise_time,args.tmax)):
        raise ValueError("All numeric parameters must be finite.")
    if min(args.kp,args.ki,args.kv) < 0 or min(args.dt,args.duration,args.rise_time,args.tmax) <= 0:
        raise ValueError("Gains must be nonnegative; time settings and force limit must be positive.")
    try:
        import opensim as osim
        # Keep missing-mesh warnings visible without printing every integrator step.
        osim.Logger.setLevelString("Warn")
        import matplotlib
        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(f"Missing package: {exc}. In your working Python environment install opensim and matplotlib.") from exc
    root = Path(__file__).resolve().parent
    source = find_input(root,args.input)
    args.live = not (args.no_live or args.no_show)
    args.plt = plt
    print(f"Python: {sys.executable}",flush=True)
    dest, number = root / "control_results", 2
    while dest.exists():
        dest = root / f"control_results_{number}"
        number += 1
    dest.mkdir()
    args.results_dir = dest
    args.opensim_version = osim.GetVersion()
    try:
        prepare_display(osim, source, args.geometry, args.live)
    except Exception as exc:
        viewer_error(args, "display setup", exc)
        args.live = False
        print("Continuing without 3D; data, final plot windows, and upload remain enabled.", flush=True)
    print(f"ARM26 PI TRACKING + GITHUB - COMPLETE RUNNER v4\nInput: {source}\nResults: {dest}", flush=True)
    print("First: comparison without integral. Next: PI simulation with automatic live 3D display." if args.live
          else "Display disabled; computing and saving results.",flush=True)
    baseline, b_summary = run_trial(osim, source, dest, args, 0, "no_integral")
    controlled, c_summary = run_trial(osim, source, dest, args, args.ki, "PI")
    plot_results(plt, baseline, controlled, dest, args)
    control_diagram(plt, dest, args)
    guide = f"""ARM26 FEEDBACK CONTROL - FIRST CONTROLLER-DEVELOPMENT EXPERIMENT

SOURCE MODEL: {source.name}
SOURCE IS NOT MODIFIED. No extra handheld load is added.
Gravity is ON; shoulder is locked at 0 degrees. Elbow starts at 20 degrees.
Reference: two 20 -> 80 -> 20 degree cycles; each leg lasts {args.rise_time} s, then hold at 20.
These angles and gains are example design choices, NOT the paper's experiment.

WHAT TO OPEN
By default Python automatically opens the native OpenSim/Simbody 3D viewer and a
live angle/error/tension dashboard during the PI run (after the comparison run).
The displayed states come directly from the force-driven simulation.
After completion, click Replay / Pause in the dashboard to review the same motion.
This viewer is a separate window, not the full OpenSim editing application.
The PI changes cable tension; it does not modify the reference trajectory.
--no-live disables live windows; --no-show disables ALL windows.
If geometry cannot be found automatically, choose its folder when prompted,
or pass --geometry "PATH_TO_GEOMETRY". The input model is searched beside this
script, then one and two folders above when --input is omitted.

AUTOMATIC GITHUB PUSH
Enabled by default for {GITHUB_REPO_URL}, branch {GITHUB_BRANCH}.
After a complete simulation, a background upload saves a dated snapshot under
opensim_runs. It includes data, motion/states, plots, input/playback models, code
and settings. Geometry assets are not uploaded. Git credentials on your computer
are required; the ChatGPT GitHub connection does not sign in your Windows Git.
The upload uses an isolated temporary clone. It does not modify your existing
repositories, change global Git settings, or force-push. A rejected upload keeps
all results locally. Read github_push_status.txt for the status and retry command.
Use --push-results "PATH_TO_RESULTS_FOLDER" to retry without repeating simulation.
Use --no-push to keep a run local. Do not close VS Code until upload status appears.

OPTIONAL SAVED FILES
control_plots.png (or PDF): six plots of tracking, input, torque and muscle response.
controller_diagram.png: feedback loop and division of work.
In OpenSim: open arm26_control_playback.osim, then File > Load Motion > PI_motion.mot.
Click Play. The arm flexes, cuffs follow their bodies, cable and muscle paths change length.
This plays recorded motion, not a live controller. To change gains and rerun, use Python.
The XML alone does NOT contain the Python feedback controller. Do not use the GUI
Simulate button expecting this Python experiment to run.
For comparison load no_integral_motion.mot onto the same playback model.

CONTROLLER
e = reference angle - actual angle (radians)
tau_raw = Kp*e + Ki*integral(e dt) - Kv*angular_velocity
T = clip(tau_raw / cable_moment_arm, 0, TENSION_MAX)
Kp={args.kp} N m/rad; Ki={args.ki} N m/(rad s); Kv={args.kv} N m s/rad.
TENSION_MAX={args.tmax} N; sample period={args.dt} s.
Kp responds to current error. Ki learns the sustained torque needed against gravity.
Kv damps movement. With Kv>0 this is PI PLUS velocity feedback (PID-style), not pure PI.
Set KV=0 or --kv 0 to investigate pure PI; inertia can cause oscillation or instability.
The no_integral trial uses the same Kp and Kv with Ki=0, for a controlled comparison.
Integral action freezes when the error would push the command further into saturation.
Only a flexor cable exists: it pulls but cannot push. Gravity assists lowering.
This does not provide arbitrary bidirectional tracking; an extensor cable is a later step.
No added joint damping or hidden gravity feedforward is used.

PYTHON AND OPENSIM
Python reads elbow angle/speed, calculates error and tension, and holds that tension
for one sample interval. OpenSim computes muscle forces and integrates the coupled
equations of motion; the resulting angle is measured again. Motion is simulated,
not imposed by prescribing the desired coordinate trajectory.
The ideal cable uses ScalarActuator.overrideActuation/setOverrideActuation in N.
This bypasses motor dynamics and actuator control scaling. A fresh Manager is used
after each input update. The 100-Hz loop is simulated time, not real-time hardware.

MUSCLES
All six Thelen muscles remain enabled. Each receives FIXED excitation {EXCITATION}.
Muscle activation starts at that same value; lengths and forces evolve with motion.
The plots show BIClong and TRIlong; CSV contains activation, fiber length, total
musculotendon path length and tendon force for all six muscles.
The .mot animates muscle paths geometrically; it does NOT encode changing excitation
or detailed muscle bulging. The .sto contains the computed states for later analysis.
These trials are NOT evidence of reduced human effort: no human command generator
or matched voluntary movement task has been implemented.

RELATION TO THE PAPER
Sambhav et al. (2022), DOI 10.3389/frobt.2022.768841, use a gravity-compensation
assistive controller plus a muscle-command framework. Their example is NOT PI.
Our example uses Python instead of MATLAB and develops angle-feedback control.
Still missing for paper replication: muscle-command generation, second cable,
paper trajectories/loads, matching geometry, and joint/strap reaction analysis.
The cuffs are visual only, not deformable tissue or contact mechanics.

RESULTS
{b_summary}
{c_summary}
RMSE includes startup and motion. A small final error does not alone prove stability.
CSV includes the saturation flag. States are in SI units; .mot angles are in degrees.
The final CSV command is logged at the endpoint and is not integrated further.

GAIN EXPERIMENTS (change one at a time; original results are preserved)
1. Compare the two default runs to see integral action reduce the holding offset.
2. Reduce KI to see slower offset correction.
3. Increase KP moderately and inspect tracking, tension and oscillations.
4. Default KV=0 is pure PI. Try KV=3 for optional velocity feedback to damp motion.
   The script stops if the angle leaves the checked 0-90 degree range.
5. Change RISE_TIME from 3 to 4 s for slower tracking. Duration adjusts automatically.
   Faster motion can exceed what this single pulling cable can achieve.

API documentation:
https://opensim-org.github.io/opensim-moco-site/docs/1.3.0/html_user/classOpenSim_1_1ScalarActuator.html
https://opensim-org.github.io/opensim-moco-site/docs/1.3.0/html_user/classOpenSim_1_1Manager.html
"""
    (dest / "README.txt").write_text(guide, encoding="utf-8")
    # Start after files are complete and BEFORE blocking plt.show().
    # A non-daemon worker finishes even if the plot windows are closed early.
    save_run_snapshot(source, dest, args, {"no_integral":b_summary, "PI":c_summary})
    upload_worker = start_github_push(dest, args)
    print(f"\nDONE. Saved plots: {dest / 'control_plots.png'}")
    if hasattr(args,"display"):
        print("Use Replay / Pause in the live dashboard to show the motion again.\n"
              "Close the plot windows when finished. No manual OpenSim loading is needed.")
    else:
        print("Optional manual animation: arm26_control_playback.osim + PI_motion.mot.")
    if not args.no_show:
        try:
            plt.show()
        except Exception as exc:
            print(f"Plot window unavailable ({exc}); the PNG and PDF are already saved.")
    if hasattr(args,"display"):
        args.display.close()
    plt.close("all")
    if upload_worker:
        if upload_worker.is_alive():
            print("Waiting for GitHub upload; see github_push_status.txt for progress...", flush=True)
        upload_worker.join()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
