LATEST UPLOAD FIX: Read STANDARD_GIT_UPLOAD_README.txt. Upload your saved complete_20261103_103737_d07f79 run with upload_results.py. This release uses ordinary git add/commit/push and verifies the manifest before and after push. No simulation rerun is needed.

VERIFIED UPLOAD UPDATE: Read UPLOAD_VERIFICATION_README.txt. Run upload_results.py alone to upload your latest saved results without retraining. GitHub API confirmation is required before the code reports verified success.

NEW: Read EDITABLE_REPORT_README.txt. generate_overleaf.py exports saved results into editable LaTeX + overleaf_figures; repeat_tests.py reruns only frozen-network OpenSim tests. run_all.py remains the full end-to-end entry point.

ARM26 COMPLETE WORKFLOW — ONE ENTRY POINT

IN VS CODE
1. Extract the ZIP.
2. Right-click the extracted Arm26_Complete folder > Open in Integrated Terminal.
3. Run:
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\run_all.py

Leave the terminal open. One command performs:
[1/5] Dataset, dependency, CUDA/CPU, OpenSim, plotting and GitHub connection checks.
[2/5] Train from saved accepted PID episodes; validate every epoch; select the best validation checkpoint; then evaluate held-out force predictions.
[3/5] Actual NN-only OpenSim tracking on EVERY accepted test episode (one shared viewer by default).
[4/5] Learning curves, force accuracy/error plots, tracking/force comparisons, tables, sampling/runtime audit, REPORT.html and OVERLEAF_REPORT.zip.
[5/5] Push the result snapshot to main in hzolfaghari2022/Continual-Learning-with-human-in-the-loop. Print the verified result URL.

DATA
Automatically finds the latest compatible collection in C:\Users\hzlfghri\Arm26_Data.
Optional exact dataset:
--dataset "C:\Users\hzlfghri\Arm26_Data\20261102_163229_UTC_51ab7d"
No data need to be moved. The full collection must be finished. Only accepted episodes are loaded, and preflight rechecks completion, motion RMSE <=2 deg, full maximum error <=5 deg. Each episode stays in its original train/validation/test split; test data never update weights or select the checkpoint. Preflight records all teacher tracking errors in preflight.json for audit. No stricter threshold or retrospective selection is silently applied.

TRAINING
Pooled A+B direct cable-force imitation, 21–64–64–1 tanh network, Adam, up to 200 epochs, patience 25, seed 42. Validation selects the best weights. Early stopping before epoch 200 is expected when validation has not improved. CUDA is used if available. Low force error is distinct from closed-loop tracking accuracy. This workflow does not assume success or claim that the NN outperforms PID.

RESULTS
C:\Users\hzlfghri\Arm26_NN_Results\complete_<timestamp>_<id>\
Open REPORT.html locally. Upload OVERLEAF_REPORT.zip to Overleaf: New Project > Upload Project > Recompile main.tex.
The report includes short explanations and all saved result plots (remaining figures in an appendix). Its sampling audit distinguishes simulation time from computation time. Local PDF compilation is not required.

IF A LATE STAGE FAILS
Rerun the SAME run_all.py command from the SAME package. It automatically finds the latest incomplete complete-workflow run. Completed training is reused only with a matching checkpoint hash and unchanged data/configuration. Completed OpenSim episodes are reused only when the checkpoint and baseline settings hashes match. Reports and upload are retried. Completed work is not deleted. If training itself fails before its completion record, that training stage must restart; no optimizer-state continuation is claimed.
Failures are recorded in workflow_state.json. Scientific outcomes are in pipeline_status.json. WORKFLOW COMPLETE is printed only after the requested stages, including report verification and GitHub confirmation, succeed. Tracking failures remain visible and give exit code 2 even when reporting/upload succeed.

GITHUB
Git for Windows and working GitHub authentication are required. The repository already used is configured by default. HTTPS uses existing credentials; a credential-manager sign-in window may appear. Other repo paths are preserved; no force push is used. Files >32 MiB are split losslessly with a reconstruction script and SHA-256 manifest. Local originals are kept. The entire run folder and this package's code/model snapshot are uploaded; the separate Arm26_Data tree is not duplicated. See GITHUB_UPLOAD_README.txt for details.

ADVANCED OPTIONS
--no-live: no viewer or browser opening.
--no-github: explicitly keep the run local.
--new-run: start a new model instead of resuming an incomplete workflow.
--epochs N: change the training limit (default 200).
--output PATH: set the result folder explicitly.
--allow-partial and --max-episodes N: exploratory tests only; omit both for your final full run.
Do not tune the model against repeated test results; use validation for development.

VALIDATION AND LIMITS
Actual Linux OpenSim dynamics, fresh short CPU training, plot/Overleaf export, stage reuse, and Git uploads to a separate local bare repository were tested. Windows CUDA training and the previous standalone uploader have worked in your environment, but this combined release's native Windows GUI/authentication cannot be executed here. No code can guarantee successful NN accuracy, network connectivity, or a fault-free external environment. The checks are designed to catch failures early and retain completed scientific results.
