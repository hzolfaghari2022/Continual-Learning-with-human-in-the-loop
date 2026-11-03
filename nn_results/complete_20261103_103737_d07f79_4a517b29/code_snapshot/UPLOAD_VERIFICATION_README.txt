VERIFIED GITHUB UPLOAD AND CONTRIBUTION ATTRIBUTION

UPLOAD YOUR SAVED RESULTS AGAIN (NO TRAINING OR SIMULATION)
From this Arm26_Complete folder in VS Code's integrated terminal:
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\upload_results.py --run-folder "C:\Users\hzlfghri\Arm26_NN_Results\recovered_20261103_084529"
Choose another result folder if you want the newer full-pipeline run. Do not retrain just to retry upload.

The uploader reports GITHUB VERIFIED SUCCESS only after:
- Git creates or identifies a commit containing the result manifest;
- the commit is on remote main, or is an ancestor when someone concurrently advances main;
- GitHub's independent public API returns the same commit and manifest blob SHA;
- the receipt records author login, author email, commit date, and whether a new commit was created.
It uses your GitHub-provided noreply email 105059650+hzolfaghari2022@users.noreply.github.com. The connected GitHub API linked that email to hzolfaghari2022 on the existing PI commit. The code does not create empty commits just to increase contribution counts.

Local Git process environment variables for a parent repository, author identity, or old commit dates are removed for the isolated uploader process. This is defensive isolation, not a confirmed diagnosis of the earlier failed/unverified NN upload. Your global settings, saved results and repository history are not rewritten.

The contribution calendar is separate from repository upload. GitHub says qualifying contributions can take up to 24 hours to appear. Reuploading identical content may make no new commit, so it does not necessarily add contribution credit. GitHub's UI is not controlled by this code.

If Git push succeeds but API verification is blocked (network, rate limits, unavailable path), the code reports failure/unconfirmed verification and retains all local results. Do not interpret that as proof the push itself failed. Read github_upload_status.json and send the console output. Retrying this upload does not rerun the NN.

All previous run_all.py / repeat_tests.py / editable Overleaf functionality is retained and uses this strengthened uploader. The original PID dataset stays where it is.
