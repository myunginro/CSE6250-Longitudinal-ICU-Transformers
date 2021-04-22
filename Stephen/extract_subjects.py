from __future__ import absolute_import
from __future__ import print_function

import argparse
import yaml

from mimic3benchmark.mimic3csv import *
from mimic3benchmark.preprocessing import add_hcup_ccs_2015_groups, make_phenotype_label_matrix
from mimic3benchmark.util import dataframe_from_csv
from pyspark import SparkContext, SQLContext
import pyspark.sql.functions as psql

parser = argparse.ArgumentParser(description='Extract per-subject data from MIMIC-III CSV files.')
parser.add_argument('mimic3_path', type=str, help='Directory containing MIMIC-III CSV files.')
parser.add_argument('output_path', type=str, help='Directory where per-subject data should be written.')
parser.add_argument('--event_tables', '-e', type=str, nargs='+', help='Tables from which to read events.',
                    default=['CHARTEVENTS', 'LABEVENTS', 'OUTPUTEVENTS'])
parser.add_argument('--phenotype_definitions', '-p', type=str,
                    default=os.path.join(os.path.dirname(__file__), '../resources/hcup_ccs_2015_definitions.yaml'),
                    help='YAML file with phenotype definitions.')
parser.add_argument('--itemids_file', '-i', type=str, help='CSV containing list of ITEMIDs to keep.')
parser.add_argument('--verbose', '-v', dest='verbose', action='store_true', help='Verbosity in output')
parser.add_argument('--quiet', '-q', dest='verbose', action='store_false', help='Suspend printing of details')
parser.set_defaults(verbose=True)
parser.add_argument('--test', action='store_true', help='TEST MODE: process only 1000 subjects, 1000000 events.')
args, _ = parser.parse_known_args()

try:
    os.makedirs(args.output_path)
except:
    pass

sc = SparkContext(appName="extract_subjects")
sqlContext = SQLContext(sc)
patients = read_patients_table(args.mimic3_path)
admits = read_admissions_table(args.mimic3_path)
stays = read_icustays_table(args.mimic3_path)
patients_df = read_patients_table_df(args.mimic3_path, sqlContext)
admits_df = read_admissions_table_df(args.mimic3_path, sqlContext)
stays_df = read_icustays_table_df(args.mimic3_path, sqlContext)
if args.verbose:
    print('START:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

stays = remove_icustays_with_transfers(stays)
# stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
stays_df = stays_df.\
    filter((stays_df.FIRST_WARDID == stays_df.LAST_WARDID) & (stays_df.FIRST_CAREUNIT == stays_df.LAST_CAREUNIT)).\
    drop(*['FIRST_CAREUNIT', 'FIRST_WARDID', 'DBSOURCE', 'ROW_ID'])
x = stays_df.count()
if args.verbose:
    print('REMOVE ICU TRANSFERS:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

stays = merge_on_subject_admission(stays, admits)
stays_df = stays_df.join(admits_df, on=['SUBJECT_ID', 'HADM_ID'])
# x = stays_df.count()
stays = merge_on_subject(stays, patients)
stays_df = stays_df.join(patients_df, on=['SUBJECT_ID'])
x = stays_df.count()
stays = filter_admissions_on_nb_icustays(stays)

to_keep = stays_df.groupby('HADM_ID').\
    agg(psql.count('ICUSTAY_ID').alias('ICUSTAY_ID')).\
    filter("ICUSTAY_ID >= 1 and ICUSTAY_ID <= 1").\
    drop('ICUSTAY_ID')
stays_df = stays_df.join(to_keep, on='HADM_ID')
# x = stays_df.count()
if args.verbose:
    print('REMOVE MULTIPLE STAYS PER ADMIT:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

stays = add_age_to_icustays(stays)
# stays["INDATE"] = pd.to_datetime(stays["INTIME"].dt.date)
# stays["DOBDATE"] = pd.to_datetime(stays["DOB"].dt.date)
# stays['AGE'] = (stays["INDATE"].subtract(stays["DOBDATE"])).dt.days // 365.0
# # stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60.0/60.0/24.0/365.0
# stays.loc[stays.AGE < 0, 'AGE'] = 90
# stays_df.withColumn("INDATE", stays_df["INTIME"].cast(TimestampType()))
# stays_df.withColumn("DOBDATE", stays_df["DOB"].cast(TimestampType()))
# stays_df.withColumn('AGE', stays_df['INDATE'] - stays_df["DOBDATE"])
stays_df = stays_df.select("*", psql.to_date(stays_df["INTIME"]).alias("INDATE"))
stays_df = stays_df.select("*", psql.to_date(stays_df["DOB"]).alias("DOBDATE"))#.alias("DOBDATE")
stays_df = stays_df.select("*", psql.floor((psql.datediff(stays_df['INDATE'], stays_df["DOBDATE"]) / 365.0)).alias("AGE"))
                           # alias("AGE"))#.alias("AGE")
print(stays_df.filter("AGE > 250").count())
stays_df = stays_df.withColumn("AGE", psql.when(stays_df.AGE > 250, 90).otherwise(stays_df.AGE))
print(stays_df.filter("AGE < 0").count())
stays = add_inunit_mortality_to_icustays(stays)
stays = add_inhospital_mortality_to_icustays(stays)
stays = filter_icustays_on_age(stays)
if args.verbose:
    print('REMOVE PATIENTS AGE < 18:\n\tICUSTAY_IDs: {}\n\tHADM_IDs: {}\n\tSUBJECT_IDs: {}'.format(stays.ICUSTAY_ID.unique().shape[0],
          stays.HADM_ID.unique().shape[0], stays.SUBJECT_ID.unique().shape[0]))

# stays.to_csv(os.path.join(args.output_path, 'all_stays.csv'), index=False)
diagnoses = read_icd_diagnoses_table(args.mimic3_path)
diagnoses = filter_diagnoses_on_stays(diagnoses, stays)
# diagnoses.to_csv(os.path.join(args.output_path, 'all_diagnoses.csv'), index=False)
count_icd_codes(diagnoses, output_path=os.path.join(args.output_path, 'diagnosis_counts.csv'))
#
phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(args.phenotype_definitions, 'r')))
make_phenotype_label_matrix(phenotypes, stays).to_csv(os.path.join(args.output_path, 'phenotype_labels.csv'),
                                                      index=False, quoting=csv.QUOTE_NONNUMERIC)
#
# if args.test:
#     pat_idx = np.random.choice(patients.shape[0], size=1000)
#     patients = patients.iloc[pat_idx]
#     stays = stays.merge(patients[['SUBJECT_ID']], left_on='SUBJECT_ID', right_on='SUBJECT_ID')
#     args.event_tables = [args.event_tables[0]]
#     print('Using only', stays.shape[0], 'stays and only', args.event_tables[0], 'table')

subjects = stays.SUBJECT_ID.unique()
# break_up_stays_by_subject(stays, args.output_path, subjects=subjects)
break_up_diagnoses_by_subject(phenotypes, args.output_path, subjects=subjects)
items_to_keep = set(
    [int(itemid) for itemid in dataframe_from_csv(args.itemids_file)['ITEMID'].unique()]) if args.itemids_file else None
for table in args.event_tables:
    read_events_table_and_break_up_by_subject(args.mimic3_path, table, args.output_path, items_to_keep=items_to_keep,
                                              subjects_to_keep=subjects)
