# possession-mtah2027

Frame-level ball possession labelling for the More Than A Hack 2027 Initial Challenge. For each of the 199,319 frames of FC Barcelona vs Athletic Club the pipeline writes one label (H, A or D) from the provided automatic tracking data (ATD).

All models and parameters are trained or tuned on external matches only. The match data is not part of this repository. Method, training data, results and limitations are described in docs/technical_document.md.

## Requirements

- Python 3.10
- The challenge data bundle, with tactical/BAR-ATH_tactical_atd.xml and tactical/BAR-ATH_tactical_atd.txt

## Reproduce the submission file

Linux and macOS:

    python3.10 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.lock
    export PYTHONPATH=src OMP_NUM_THREADS=1 PYTHONHASHSEED=0
    python -m possession.run --data-dir /path/to/challenge-data --out possession.csv
    python /path/to/more-than-a-hack-2027/evaluation/validate_submission.py possession.csv

Windows:

    setup_windows.bat
    run_windows.bat DATA_DIR OUT_DIR VALIDATOR_SCRIPT

setup_windows.bat creates the environment and runs the tests. run_windows.bat runs the pipeline twice, compares the two outputs byte by byte and runs the official validator.

possession.run writes the CSV and a .sha256 file next to it. Model files are checked against the hashes in config/frozen.json before use. The run needs NumPy and scikit-learn only.

## Retrain the models

The models are trained on the IDSSE open data (Bassek et al. 2025, CC BY 4.0) and the frame-synchronised events published with PathCRF (MPL 2.0), both downloaded separately. Retraining the temporal networks needs PyTorch (requirements-train.txt). With PYTHONPATH=src, a cache folder CACHE and NAMES standing for the 35 variants J03WMX_v00 to J03WR9_v04:

    python -m possession.external.build --idsse-dir IDSSE --synced-dir SYNCED --cache-dir CACHE
    python -m possession.external.make_degraded --cache-dir CACHE --match J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9 --variants 0 1 2 3 4
    python -m possession.external.make_features --cache-dir CACHE --names NAMES
    python -m possession.external.make_features --cache-dir CACHE --names NAMES --randomized
    python -m possession.external.train --cache-dir CACHE --final --model-dir models --with-randomized
    python -m possession.external.train_temporal --cache-dir CACHE --label y --final --model-dir models --with-randomized
    python -m possession.external.train_temporal --cache-dir CACHE --label y1 --final --model-dir models --with-randomized

Held-out evaluation uses --fold MATCH instead of --final, then eval_decoder and select_decoder for the nested decoder selection. After retraining, update the hashes in config/frozen.json.

## Tests

    pip install -r requirements-dev.lock
    PYTHONPATH=src python -m pytest -q tests

Two tests need environment variables: MTAH_IDSSE_DIR (attack direction against DFL metadata) and MTAH_E2E=1 (full run on a synthetic bundle).

## Layout

- src/possession: pipeline (io_atd, ball, features, features_control, temporal, decoder, kickoff, run)
- src/possession/external: label construction, degradation, randomization, features, training and evaluation on external matches
- src/possession/vendor: official scorer with its MIT licence (LICENSE_mtah2027)
- config: decoder parameters and model hashes
- models: trained models
- docs: technical document
- tests: unit and end-to-end tests

## Licence

MIT. External datasets and their licences are listed in docs/technical_document.md.
