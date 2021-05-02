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
1. put the main.py inside "<your-path>\CSE6250-Longitudinal-ICU-Transformers\<your-name>\mimic3models\in_hospital_mortality"
2. in terminal, cd to this location "<your-path>\CSE6250-Longitudinal-ICU-Transformers\<your-name>"
3. run this command in terminal " python -um mimic3models.in_hospital_mortality.main --network <whatever>"


## Training
       python -um mimic3models.in_hospital_mortality.main --network <whatever>
