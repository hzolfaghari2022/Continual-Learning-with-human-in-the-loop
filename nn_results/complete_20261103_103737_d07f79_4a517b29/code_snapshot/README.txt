ARM26 — ONE COMMAND, REUSE YOUR TRAINED MODEL

VS CODE
Extract this ZIP. Right-click the Arm26_Complete folder containing run_all.py > Open in Integrated Terminal.
Run:
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\run_all.py

DEFAULT BEHAVIOR — DOES NOT TRAIN
1. Find your saved trained result (latest full-pipeline pointer, or latest completed saved evaluation).
2. Reuse the trained checkpoint and recorded tests; if exports or tests are missing, regenerate them with frozen weights, never fit a new model.
3. Generate training/validation/test figures, tracking plots and reports.
4. Verify that every LaTeX figure call resolves and every required source/image is present in OVERLEAF_REPORT.zip.
5. Open REPORT.html and, on Windows, open the actual overleaf_report folder in File Explorer.
6. Upload all saved results/reports using the standard Git uploader and verify the remote result manifest.
The exact source result, report filenames, figure folder and ZIP are printed.

TO SELECT YOUR KNOWN COMPLETED RUN EXPLICITLY
Append:
--run-folder "C:\Users\hzlfghri\Arm26_NN_Results\complete_20261103_103737_d07f79"
The default one-command workflow also finds that latest full-pipeline result automatically if its pointer is available.

WHERE THE OVERLEAF REPORT IS
Inside YOUR saved result folder:
  overleaf_report/main.tex
  overleaf_report/results.tex
  overleaf_report/overleaf_figures/learning_curve.png
  overleaf_report/overleaf_figures/test_parity.png
  overleaf_report/overleaf_figures/tracking_<episode>_comparison.png
  ... every saved figure
  OVERLEAF_REPORT.zip
Upload OVERLEAF_REPORT.zip to Overleaf: New Project > Upload Project. Compile main.tex. Edit results.tex for text, captions and figure calls. To use an existing project, copy overleaf_figures and results.tex, add graphicx,float,booktabs,longtable,amsmath packages as needed, then use \input{results} or copy individual figure environments. No local LaTeX installation is required. Keep your own edits separately before re-exporting local generated sources.

TRAIN A NEW MODEL ONLY WHEN YOU INTEND TO
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\run_all.py --train-new
This explicitly starts a new model, trains/validates, tests every accepted test episode in OpenSim, creates reports and uploads. Training uses CUDA if available, up to 200 epochs with early stopping patience 25, and selects the best validation checkpoint. Test episodes never select weights. Without --train-new, fitting is never called.

REPEAT ONLY TESTS FOR YOUR ADVISOR
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\repeat_tests.py
Reruns the frozen network in OpenSim with a live viewer, keeps original outputs, and creates separate results/reports/uploads. Append --max-episodes 3 for a shorter demonstration, or --run-folder "FULL_RESULT_FOLDER" to choose the saved model. Same deterministic conditions generally reproduce the same trajectories; repeated tests are a demonstration, not new learning evidence.

OPTIONS
--no-github: keep outputs local.
--no-live: do not open the browser/Explorer or a test viewer.
--run-folder PATH: choose the existing run explicitly.
--epochs N and --dataset PATH: apply to explicit new training.

FAILURES
Reports open before upload, so an upload problem does not hide the Overleaf output. Existing results and weights are preserved. Read report_console.log or github_console.log in the saved run folder. Retry run_all.py to regenerate/reupload without training. If no model is found, it stops and requests the correct saved run path; it does not silently train from scratch.
No successful tracking outcome is assumed. Failed tests remain visible in reports. Code cannot guarantee network connectivity, Windows dependencies, or controller accuracy.
