UPLOAD FIX FOR THE MISSING MANIFEST ERROR

Your completed run is saved at:
C:\Users\hzlfghri\Arm26_NN_Results\complete_20261103_103737_d07f79
No training, validation, testing, or reporting needs to be repeated to fix upload.

Open THIS Arm26_Complete folder in VS Code's integrated terminal and run:
& "C:\Users\hzlfghri\miniconda3\envs\lstm-py312\python.exe" .\upload_results.py --run-folder "C:\Users\hzlfghri\Arm26_NN_Results\complete_20261103_103737_d07f79"

This version replaces the low-level index-info/read-tree/commit-tree construction with an ordinary isolated Git clone and working tree:
1. Clone current main.
2. Copy saved outputs into the generated nn_results/<run_id> folder.
3. git add --force --all, then verify the staged manifest and its blob hash.
4. git commit, confirm the new commit contains the manifest, then git push without force.
5. Confirm the remote branch and query GitHub's API for the committed manifest.

The error you supplied proves that the old PI commit was being checked without the NN manifest. The precise reason the earlier low-level path did not create the expected tree on your computer was not established; this update removes that path entirely. The new upload prints five progress stages and fails before claiming success if the manifest is absent.

Only the uploader is changed. Scientific data, model architecture, learned weights, trajectories and reports are unchanged. Local results and other repository folders are preserved. Files above 32 MiB remain losslessly split and include a restoration script. The code still uses the GitHub-linked noreply address. No empty contribution-only commits are created.

Validation: initial upload to a local bare Git remote, remote manifest/commit inspection, unchanged retry, changed output commit, unrelated-file preservation, Windows CRLF file-byte preservation with repository text attributes, large-file reassembly, and verification failure handling all passed. GitHub API response validation was tested with fixtures; native Windows Git and the actual full-result upload are not executable here.
