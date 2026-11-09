"""One persistent OpenSim 3D viewer per collector; compatible drop-in patch.
Episode workers only publish actual states. They never create native viewers.
The viewer process exits when the collector exits. No extra terminal is opened.
"""
import runtime_setup
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    # Retry sharing conflicts from Windows file scanning without losing physics.
    for attempt in range(5):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(.01)


def process_alive(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class EpisodeViewer:
    def __init__(self, osim, model, state, root):
        self.physics = model
        worker_index=sys.argv.index('--worker') if '--worker' in sys.argv else -1
        if worker_index<0:raise RuntimeError('Live plots require the collector episode worker.')
        episode_folder=Path(sys.argv[worker_index+1])
        self.cfg=json.loads((episode_folder/'settings.json').read_text(encoding='utf-8-sig'))
        self.elbow=model.getCoordinateSet().get('r_elbow_flex')
        self.cable=osim.PathActuator.safeDownCast(model.getForceSet().get('exo_flexor'))
        # Each episode worker is a direct child of the same collection process.
        owner = os.getppid()
        self.session = Path(tempfile.gettempdir()) / ('arm26_live_' + str(owner))
        self.session.mkdir(exist_ok=True)
        self.next_frame = 0.
        self.started = time.perf_counter()
        self.names = model.getStateVariableNames()
        self.names = [self.names.get(i) for i in range(self.names.getSize())]
        launch = self.session / 'launched.json'
        ready = self.session / 'ready.json'
        if not launch.exists():
            # Mark BEFORE launch. A failed/closed viewer must not spawn 200 windows.
            atomic_json(launch, {'owner': owner})
            model.printToXML(str(self.session / 'display_model.osim'))
            with (self.session / 'viewer.log').open('a', encoding='utf-8') as log:
                subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), '--serve',
                     str(self.session), str(owner), str(Path(root).resolve())],
                    stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                    close_fds=True)
        deadline = time.monotonic() + 30
        while not ready.exists():
            if (self.session / 'error.txt').exists():
                raise RuntimeError((self.session / 'error.txt').read_text())
            if time.monotonic() >= deadline:
                raise RuntimeError('Viewer startup timed out. See ' + str(self.session / 'viewer.log'))
            time.sleep(.1)
        metadata = json.loads(ready.read_text())
        if not process_alive(metadata['pid']):
            raise RuntimeError('Shared viewer has closed; no automatic replacement will be opened.')
        if metadata['state_names'] != self.names:
            raise RuntimeError('Display and simulation state layouts differ.')
        self.update(state, force=True)
        print('SHARED VIEWER: reusing one window for the whole collection.', flush=True)

    def update(self, state, force=False):
        t = float(state.getTime())
        if not force and t + 1e-9 < self.next_frame:
            return
        self.next_frame = t + 1/30
        values = self.physics.getStateVariableValues(state)
        atomic_json(self.session / 'frame.json', {
            'sequence': time.time_ns(), 'worker': os.getpid(), 'time_s': t,
            'values': [values.get(i) for i in range(values.size())],
            'angle_deg': math.degrees(self.elbow.getValue(state)),
            'tension_N': float(self.cable.getActuation(state)),
            'config': {key:self.cfg[key] for key in ['episode_id','task','trajectory','shoulder_deg',
                'mass_kg','start_hold_s','motion_duration_s','trajectory_time_scale','duration_s','force_cap_N']}})
        delay = t - (time.perf_counter() - self.started)
        if delay > 0:
            time.sleep(min(delay, .05))

    def close(self):
        # Retain exactly one viewer across episode boundaries and uploads.
        pass


class LiveDashboard:
    """One persistent plot window. Histories reset for each episode."""
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt=plt
        plt.ion()
        self.fig,self.axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        self.fig.subplots_adjust(top=.86,bottom=.09,hspace=.22)
        self.title=self.fig.suptitle('Waiting for simulation...',fontsize=12)
        self.desired,=self.axes[0].plot([],[],'k--',lw=2,label='Desired trajectory')
        self.actual,=self.axes[0].plot([],[],color='#1864ab',lw=2,label='Actual OpenSim angle')
        self.axes[0].set_ylabel('Elbow angle (deg)');self.axes[0].legend(loc='upper right')
        self.error,=self.axes[1].plot([],[],color='#d9480f',lw=1.5)
        self.axes[1].axhline(0,color='gray',lw=.8);self.axes[1].set_ylabel('Reference - actual (deg)')
        self.force,=self.axes[2].plot([],[],color='#2b8a3e',lw=1.5)
        self.axes[2].set(ylabel='Cable tension (N)',xlabel='Episode time (s)')
        for ax in self.axes:ax.grid(alpha=.25)
        self.cursor=self.axes[0].axvline(0,color='gray',alpha=.6)
        self.episode=None
        self.times=[];self.angles=[];self.errors=[];self.forces=[]
        plt.show(block=False);plt.pause(.05)

    def alive(self):return self.plt.fignum_exists(self.fig.number)

    def update(self,frame):
        from trajectories import reference
        import numpy as np
        cfg=frame['config'];episode=(frame['worker'],cfg['episode_id'])
        if episode!=self.episode:
            self.episode=episode
            self.times=[];self.angles=[];self.errors=[];self.forces=[]
            ts=np.linspace(0,cfg['duration_s'],600)
            qs=[math.degrees(reference(t,cfg)[0]) for t in ts]
            self.desired.set_data(ts,qs)
            self.axes[0].set_xlim(0,cfg['duration_s'])
            self.axes[0].set_ylim(min(qs)-8,max(qs)+8)
            self.axes[2].set_ylim(0,cfg['force_cap_N']*1.05)
        t=frame['time_s'];actual=frame['angle_deg']
        desired=math.degrees(reference(t,cfg)[0])
        self.times.append(t);self.angles.append(actual);self.errors.append(desired-actual);self.forces.append(frame['tension_N'])
        self.actual.set_data(self.times,self.angles)
        self.error.set_data(self.times,self.errors);self.force.set_data(self.times,self.forces)
        limit=max(2,max(abs(v) for v in self.errors)*1.15)
        self.axes[1].set_ylim(-limit,limit)
        low,high=self.axes[0].get_ylim()
        self.axes[0].set_ylim(min(low,actual-3),max(high,actual+3))
        self.cursor.set_xdata([t,t])
        self.title.set_text(f"{cfg['episode_id']} | {cfg['trajectory']} | shoulder {cfg['shoulder_deg']:g} deg | {cfg['mass_kg']:.2f} kg\n"
            f"Time {t:.2f} s | error {desired-actual:+.2f} deg | cable {frame['tension_N']:.1f} N")
        self.fig.canvas.draw_idle()

    def pump(self):self.plt.pause(.02)
    def close(self):self.plt.close(self.fig)


def serve(session, owner, root):
    native = None
    dashboard = None
    try:
        import opensim as o
        from viewer_support import prepare_display
        prepare_display(o, root / 'arm26_exosuit.osim', root / 'Geometry', True)
        model = o.Model(str(session / 'display_model.osim'))
        # Shoulder is fixed in physics, but the shared display must accept each episode's angle.
        for i in range(model.updCoordinateSet().getSize()):
            model.updCoordinateSet().get(i).setDefaultLocked(False)
        model.setUseVisualizer(True)
        state = model.initSystem()
        viewer = model.updVisualizer()
        native = viewer.updSimbodyVisualizer()
        names = model.getStateVariableNames()
        names = [names.get(i) for i in range(names.getSize())]
        viewer.show(state)
        dashboard=LiveDashboard()
        atomic_json(session / 'ready.json', {'pid': os.getpid(), 'state_names': names})
        last = None
        while process_alive(owner) and dashboard.alive():
            try:
                frame = json.loads((session / 'frame.json').read_text(encoding='utf-8'))
            except (OSError, ValueError):
                dashboard.pump()
                continue
            if frame['sequence'] != last:
                values = o.Vector(len(frame['values']), 0.)
                for i, value in enumerate(frame['values']):
                    values.set(i, value)
                state.setTime(frame['time_s'])
                model.setStateVariableValues(state, values)
                model.realizeVelocity(state)
                viewer.show(state)
                dashboard.update(frame)
                last = frame['sequence']
            dashboard.pump()
    except Exception:
        (session / 'error.txt').write_text(traceback.format_exc(), encoding='utf-8')
    finally:
        if dashboard is not None:dashboard.close()
        if native is not None:
            try:
                native.shutdown()
            except Exception:
                pass


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--serve':
        serve(Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit('This module is loaded automatically by collect_pid_data.py.')
