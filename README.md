# CSE6250-Longitudinal-ICU-Transformers

Banchmark reference: https://github.com/YerevaNN/mimic3-benchmarks


## Overview
Python suite to construct features and classification model for in-hospital mortality from the MIMIC-III clinical database.


## Requirements
Access to the MIMIC-III dataset is required, https://mimic.physionet.org/, and download the CSVs. The following modules are also required:

- Numpy
- Pandas
- PySpark
- Keras
- MatPlotLib

## File Management
Clone the benchmark reference repo, extract MIMIC-III dataset into the /data/ folder of the root. Replace three files in the benchmark as follows, with the three files in the /In_Hospital_Mortality/ folder of this repo:

- extract_subjects.py - replace corresponding file in /mimic3benchmark/scripts/
- main.py - replace corresponding file in /mimic3models/in_hospital_mortality/
- mimic3csv.py - replace corresponsing file in /mimic3benchmark/

## ETL
Process emulates the same feature selection and filtering that was used in the benchmark study, but with steps implemented in PySpark rather than in pure Pandas. The following steps will generate a processed dataset ready for training models:

       python -m mimic3benchmark.scripts.extract_subjects {PATH TO MIMIC-III CSVs} data/root/
       python -m mimic3benchmark.scripts.validate_events data/root/
       python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
       python -m mimic3benchmark.scripts.split_train_and_test data/root/
       python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
       
## Train / validation split
       python -m mimic3models.split_train_val {dataset-directory}

## Training
cd into your root folder, and run the following:
       python -um mimic3models.in_hospital_mortality.main --network 0
