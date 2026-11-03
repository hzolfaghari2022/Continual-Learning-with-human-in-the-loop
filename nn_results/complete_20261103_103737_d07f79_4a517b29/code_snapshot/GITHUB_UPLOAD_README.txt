AUTOMATIC GITHUB RESULT UPLOAD
Repository: https://github.com/hzolfaghari2022/Continual-Learning-with-human-in-the-loop
Branch: main
Remote folder: nn_results/<run_name>_<unique_id>/

UPLOAD YOUR ALREADY COMPLETED RUN IN VS CODE
From this Arm26_NN_Only folder's integrated terminal:
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\upload_results.py --run-folder "C:\Users\hzlfghri\Arm26_NN_Results\recovered_20261103_084529"
This only uploads saved files; no training or simulation starts.

FUTURE RUNS
Run train_and_report.py as before. After reports and plots are saved, the entire result folder is uploaded automatically. Failed/interrupted scientific runs can also be uploaded, explicitly labelled in their pipeline_status.json. Recovery runs also upload automatically. Upload errors are recorded separately in github_upload_status.json and do not erase outputs or change scientific status. Retry upload_results.py for the same result folder; the stable GitHub run ID is reused.

Included: policy.pt, exported inference weights, histories, split manifests, training/validation/test figures, all tracking CSV/state files and figures, logs, HTML, Overleaf source/ZIP, and a snapshot of this package's code/model files. The separate original Arm26_Data collection is not copied wholesale; it remains at its existing location and its used episodes/hashes are recorded by the training manifest.

Large files above 32 MiB are split losslessly under .large_files. The repository upload_manifest.json records original names, SHA-256 hashes and part paths. After downloading the complete run folder, run python restore_large_files.py to reconstruct and verify the large files. Ordinary plots and small files remain directly browsable on GitHub.

AUTHENTICATION
Git for Windows must be installed and available in VS Code. HTTPS is the default, using your existing Git credentials; Git Credential Manager may open a sign-in window. No password/token is stored in this package. If you prefer configured SSH, append:
--github-url "git@github.com:hzolfaghari2022/Continual-Learning-with-human-in-the-loop.git"
SSH must already be configured on your computer.

To keep a future run local, append --no-github to train_and_report.py or recover_and_report.py.
Report-only generate_overleaf.py does not upload automatically; after regenerating reports, rerun upload_results.py to update that run's upload.

SAFETY OF SAVED DATA
This uploader creates a commit on the current main and pushes without force. It modifies only its generated run folder and preserves other paths. Concurrent changes to main trigger a retry against its new head. Local original result files are never deleted. The remote snapshot excludes only Git metadata, Python cache directories and the uploader's live status file. Paths, logs and code provenance can contain your local Windows username; the repository is public.

VALIDATION
Tested against a local bare Git repository: initial push, retry, preservation of unrelated files, large-file reconstruction/hash verification, and failure status retention. The connected GitHub plugin confirmed repository access and push permission. Your Windows sign-in and full result upload still run on your computer. No preliminary/sample results were uploaded to the real repository during development.
