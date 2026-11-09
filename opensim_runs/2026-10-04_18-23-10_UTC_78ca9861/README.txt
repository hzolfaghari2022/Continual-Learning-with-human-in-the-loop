ARM26 FEEDBACK CONTROL - FIRST CONTROLLER-DEVELOPMENT EXPERIMENT

SOURCE MODEL: arm26_exosuit_with_cable_2.osim
SOURCE IS NOT MODIFIED. No extra handheld load is added.
Gravity is ON; shoulder is locked at 0 degrees. Elbow starts at 20 degrees.
Reference: two 20 -> 80 -> 20 degree cycles; each leg lasts 3.0 s, then hold at 20.
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
Enabled by default for https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git, branch main.
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
Kp=30.0 N m/rad; Ki=15.0 N m/(rad s); Kv=0.0 N m s/rad.
TENSION_MAX=80.0 N; sample period=0.01 s.
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
All six Thelen muscles remain enabled. Each receives FIXED excitation 0.01.
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
no_integral: RMSE 3.895 deg; final error 0.141 deg; peak tension 30.49 N
PI: RMSE 2.246 deg; final error -1.662 deg; peak tension 31.09 N
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
