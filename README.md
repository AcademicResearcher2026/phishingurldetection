# phishingurldetection
PHISHING URL DETECTION USING MACHINE LEARNING: AN ANALYSIS OF GENERALIZATION ACROSS INDEPENDENT DATASETS
This repository contains code and result files for the study on phishing URL detection and cross-dataset generalization.

# Environment
Python 3.12 was used.
Install dependencies:
pip install -r requirements.txt

# Data
The raw datasets are not redistributed. Download them according to data/README.md.
Expected file paths:
data/raw/phiusiil.csv
data/raw/malicious_urls.csv

# Reproduction
Run the scripts in numerical order:
python src/01_inspect_datasets.py
python src/02_prepare_datasets.py
python src/03_extract_features.py
python src/04_train_models.py
python src/05_feature_and_shift_analysis.py
python src/06_ablation_study.py
python src/07_make_paper_tables.py
python src/08_normalized_url_control_experiment.py
python src/09_make_final_tables.py

The generated outputs are saved in results/, figures/ and paper/.
