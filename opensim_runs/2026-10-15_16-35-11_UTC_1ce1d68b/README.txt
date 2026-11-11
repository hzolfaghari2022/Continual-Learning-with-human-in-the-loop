Compare constant loads with the same PI gains, motion and initial states.

Each load starts a fresh independent OpenSim trial. Default figure overlays
0.5/2/5 kg at the original experiment's 120 N cap. A separate 240 N capacity
check tests whether force saturation explains the failure. No gain adaptation.

FAIR OVERLAID FIXED-PI LOAD COMPARISON

python .\compare_fixed_PI_loads.py --no-push

The figure overlays separate 0.5, 2 and 5 kg trials. Every trial has the SAME
Kp=30, Ki=15, reference trajectory, initial angle/speed/muscle states, fixed
muscle excitation, dt, integration accuracy and duration. Only physical load
mass and its associated inertia change within each capacity group.
Each load is constant during its independent trial: this avoids confusing
different masses with different phases of a trajectory or earlier load history.

The left column uses the previous 120 N cable limit. The right column explicitly
tests a different, hypothetical 240 N limit to help distinguish actuator capacity
from fixed-gain dynamic performance. The PI gains remain unchanged in BOTH columns.
The 240 N limit is a diagnostic design choice, not a measured device capability.
Use --capacity-check 0 to show only the 120 N load comparison.

All-load plots: target/actual angle, error and commanded tension on common scales.
comparison_metrics.csv: full-trial RMSE and second-half RMSE, upper and lower
saturation reported separately, peak error, completion status and failure details.
Failed trials are retained and plotted only until they fail; their partial RMSE
is not comparable with a full-trial score. No missing interval is invented.

The highest load at the primary cap opens a live viewer/dashboard after earlier
cases compute. Final overlaid plots appear when the comparisons finish.
--no-live skips the live viewer; --no-show computes without any windows.

Current is not modeled. Commands are ideal cable tension in newtons. The arm
still has only a flexor cable and fixed low muscle excitation; this is an
angle-feedback control experiment, not a model of a person's voluntary effort.
Do not infer that ALL fixed PI designs must fail under any heavier load.
Whether retuning helps must be tested at adequate actuator capacity, alongside
a defined error tolerance and independent trajectories/load sequences.

GitHub push uses the same repository and credentials as before. --no-push keeps
the run local. Retry with compare_fixed_PI_loads.py --push-results PATH.


AUTOMATIC COMBINED CONSTANT + CHANGING LOAD FIGURE
The comparison runner now computes an additional changing-load trial and saves
all_scenarios_fixed_PI.png and all_scenarios_fixed_PI.pdf automatically. It shows
all constant-load cases at the primary cap plus the changing-load trace together
in mass, motion, error and tension panels. The changing-load trial opens the live
viewer; constant-load cases compute first. Default events are 16.5/32.5/48.5 s.
Changes divide the movement interval into four parts if timing/cycles are edited.

Replot a completed NEW result folder without running physics:
python .\plot_combined_scenarios.py "PATH_TO_RESULTS_FOLDER"
New folders contain changing_load/fixed_PI_data.csv and schedule metadata.
Older folders without that additional run cannot reconstruct the combined figure.

Run only the primary comparison plus changing-load trial (four simulations):
python .\compare_fixed_PI_loads.py --capacity-check 0 --no-push
The combined PNG/PDF is generated and displayed by the code itself.
