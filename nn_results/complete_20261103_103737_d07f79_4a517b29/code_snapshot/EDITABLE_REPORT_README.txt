EDITABLE OVERLEAF REPORT AND TEST REPEATS

EXPORT SAVED RESULTS (NO TRAINING OR SIMULATION)
Open this Arm26_Complete folder in VS Code's integrated terminal and run:
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\generate_overleaf.py
It selects the latest complete-pipeline result and prints its path. To choose a specific result, append --run-folder "FULL_RESULT_FOLDER".

The saved run's overleaf_report folder contains:
  main.tex
  results.tex
  overleaf_figures/learning_curve.png
  overleaf_figures/test_parity.png
  overleaf_figures/test_residuals.png
  overleaf_figures/tracking_<episode_id>_comparison.png
  ... all other saved PNGs with stable descriptive names.
An OVERLEAF_REPORT.zip is also generated in the run folder.

NEW OVERLEAF PROJECT
New Project > Upload Project > select OVERLEAF_REPORT.zip. Compile main.tex. Edit results.tex to change explanations, captions, tables and figure calls.

YOUR EXISTING OVERLEAF PROJECT
Copy overleaf_figures and results.tex into your project. Add packages graphicx, float, booktabs, longtable, amsmath to its preamble if absent; insert \input{results} at the desired position. Or copy individual figure calls, for example:
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{overleaf_figures/learning_curve.png}
\caption{Training and validation force prediction error.}
\label{fig:learning_curve}
\end{figure}
Re-exporting regenerates the local LaTeX files; preserve your edits separately. Your online Overleaf project is not overwritten by this code.

REPEAT ONLY FROZEN-NETWORK OPENSIM TESTS
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\repeat_tests.py
It uses your saved checkpoint, reruns every accepted test episode with the live viewer, and creates a new test_repeat_<timestamp>_<id> folder under C:\Users\hzlfghri\Arm26_NN_Results. It generates new reports and uploads the results to the configured GitHub repository. Original outputs remain unchanged. It never invokes training or updates weights. Original learning curves and force-imitation evidence are reused and clearly identified in the report.
Options:
--run-folder "FULL_RESULT_FOLDER": choose the saved model explicitly.
--max-episodes 3: shorter demonstration with three test episodes (a subset).
--no-live: no live viewer.
--no-github: keep the repeat local.
Repeating the same settings generally reproduces similar results. This is a demonstration/reproducibility check, not an independent generalization experiment. Do not tune weights against the test set.

FULL WORKFLOW
Run run_all.py for new training, validation, all tests, figures, Overleaf report and GitHub upload. Its automatic Overleaf export now uses overleaf_figures and results.tex.
