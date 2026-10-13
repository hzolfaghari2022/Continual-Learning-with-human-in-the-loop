Time-varying handheld load: fixed PI versus manually scheduled PI.

Loads change at timed exchange events. Each event rebuilds physical mass/inertia
and transfers continuous joint/muscle state. This assumes load exchange at the
hand's current velocity without an impact impulse. It is not a variable-mass
flow model or an automatic/continually learning controller.


TIME-VARYING LOAD EXPERIMENT

Run from this extracted project using the working lstm-py312 interpreter:
python .\run_variable_load.py --no-push
The fixed-gain comparison runs first. Then the scheduled-gain run opens the
native 3D viewer and live plots, followed by final comparison plots.
Use --no-show --no-push for headless computation. Use --no-live for final plots only.

Edit variable_load_schedule.csv:
time_s is the event time; load_kg is the new dumbbell mass; kp/ki are the manually
chosen scheduled gains. Fixed PI uses --kp and --ki throughout. Both see the same
load history, motion, fixed muscle excitation and cable limit. For 4-second motion
legs, changes at 8.5/16.5/24.5 s coincide with bend-and-return cycle boundaries.
Repeat the earlier 0.5 kg condition at the end to create an A-B-C-A task sequence.

Load changes are physical mass AND inertia changes, not just extra gravity force.
At each timed event, the code rebuilds the OpenSim model, transfers joint positions,
velocities, muscle activations and fiber states, and resumes integration at the same
time. It preserves the integral contribution across load changes. At scheduled
gain changes it adjusts integral torque for a bumpless change in the raw PI command.
No target motion is imposed on the actual joint.

Interpretation: a load is exchanged for another mass, with ideal rigid grip and
no impact impulse (the new load shares current hand velocity). This is a simplified
piecewise-load scenario, not a dumbbell that physically changes density continuously.
Changing model mass in an existing State without rebuilding is not used.

The dumbbell keeps the same visible size. Its changing mass is printed in the
terminal and live dashboard; the final first plot shows exact mass versus time.
Viewer/replay shows joint and muscle motion. Stored replay geometry is fixed-size;
the playback model's inertial mass is the final load. A motion replay does not
re-execute the load changes; rerun Python to reproduce the force-driven experiment.

Outputs include load, Kp, Ki, angle, target, error, tension, PI contributions,
saturation, muscle response and states. segment_metrics.csv reports errors per
load interval, so a good overall average does not hide poor heavy-load tracking.

Default cable capacity is explicitly 120 N rather than the previous 80 N to help
separate tuning sensitivity from an actuator limit. It is a chosen simulation
parameter, not a measured exosuit rating. To test the original limit:
python .\run_variable_load.py --tmax 80 --no-push
You cannot fix insufficient cable capacity merely by increasing PI gains.
Only a flexor cable exists, so faster lowering is constrained by gravity and
passive/muscle mechanics. The 0-90 degree geometry check still stops invalid runs.

The scheduled gains are a prescribed comparison, not automatically adapted or
continually learned gains. A future continual learner could choose gains or add
a tension correction, with fixed PI and this manual schedule as baselines.
Do not claim all variable loads make fixed PI fail: inspect actual errors and
capacity limits. These example gain values are not globally optimized.

GitHub uploading uses your existing repository and local credentials. --no-push
keeps a run local. Upload retries use this runner:
python .\run_variable_load.py --push-results "PATH_TO_VARIABLE_LOAD_RESULTS"
The uploaded results contain models and scripts but not bone geometry assets.

Electrical current remains unmodeled. The commanded input is ideal cable tension.
