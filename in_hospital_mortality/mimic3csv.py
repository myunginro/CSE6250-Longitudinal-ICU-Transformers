from __future__ import absolute_import
from __future__ import print_function

import csv
import numpy as np
import os
import pandas as pd
from pyspark.sql.types import TimestampType
from tqdm import tqdm

from mimic3benchmark.util import dataframe_from_csv
from pyspark import SparkContext, SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp
import pyspark.sql.functions as psql
import sys

def read_patients_table(mimic3_path):
    pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    pats.DOB = pd.to_datetime(pats.DOB)
    pats.DOD = pd.to_datetime(pats.DOD)
    return pats

def read_patients_table_df(mimic3_path, sqlContext):
    pats = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").\
        option("mode", "DROPMALFORMED")\
        .load(os.path.join(mimic3_path, 'PATIENTS.csv')).\
        select('SUBJECT_ID', 'GENDER', to_timestamp('DOB').alias('DOB'), to_timestamp('DOD').alias('DOD'))
    # print(data.show())
    # print("hi")
    # pats = dataframe_from_csv(os.path.join(mimic3_path, 'PATIENTS.csv'))
    # pats = pats[['SUBJECT_ID', 'GENDER', 'DOB', 'DOD']]
    # pats.DOB = pd.to_datetime(pats.DOB)
    # pats.DOD = pd.to_datetime(pats.DOD)
    # print(pats)
    return pats


def read_admissions_table(mimic3_path):
    admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits

def read_admissions_table_df(mimic3_path, sqlContext):
    admits = sqlContext.read.format("com.databricks.spark.csv").option("header", "true"). \
        option("mode", "DROPMALFORMED") \
        .load(os.path.join(mimic3_path, 'ADMISSIONS.csv')). \
        select('SUBJECT_ID', 'HADM_ID', to_timestamp('ADMITTIME').alias('ADMITTIME'), to_timestamp('DISCHTIME').alias('DISCHTIME'), to_timestamp('DEATHTIME').alias('DEATHTIME'), 'ETHNICITY', 'DIAGNOSIS')
    # admits = dataframe_from_csv(os.path.join(mimic3_path, 'ADMISSIONS.csv'))
    # admits = admits[['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS']]
    # admits.ADMITTIME = pd.to_datetime(admits.ADMITTIME)
    # admits.DISCHTIME = pd.to_datetime(admits.DISCHTIME)
    # admits.DEATHTIME = pd.to_datetime(admits.DEATHTIME)
    return admits


def read_icustays_table(mimic3_path):
    stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    stays.INTIME = pd.to_datetime(stays.INTIME)
    stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    return stays

def read_icustays_table_df(mimic3_path, sqlContext):
    stays = sqlContext.read.format("com.databricks.spark.csv").option("header", "true"). \
        option("mode", "DROPMALFORMED") \
        .load(os.path.join(mimic3_path, 'ICUSTAYS.csv'))#. \
        # select('SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ETHNICITY', 'DIAGNOSIS')
    stays = stays.withColumn("INTIME", stays["INTIME"].cast(TimestampType()))
    stays = stays.withColumn("OUTTIME", stays["OUTTIME"].cast(TimestampType()))
    # stays = dataframe_from_csv(os.path.join(mimic3_path, 'ICUSTAYS.csv'))
    # stays.INTIME = pd.to_datetime(stays.INTIME)
    # stays.OUTTIME = pd.to_datetime(stays.OUTTIME)
    # print(stays.show())
    return stays


def read_icd_diagnoses_table(mimic3_path):
    codes = dataframe_from_csv(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv'))
    codes = codes[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']]
    diagnoses = dataframe_from_csv(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.merge(codes, how='inner', left_on='ICD9_CODE', right_on='ICD9_CODE')
    diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']] = diagnoses[['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM']].astype(int)
    return diagnoses

def read_icd_diagnoses_table_df(mimic3_path, sqlContext):
    codes = sqlContext.read.format("com.databricks.spark.csv").option("header", "true"). \
        option("mode", "DROPMALFORMED") \
        .load(os.path.join(mimic3_path, 'D_ICD_DIAGNOSES.csv')).\
        select('ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE')
    diagnoses = sqlContext.read.format("com.databricks.spark.csv").option("header", "true"). \
        option("mode", "DROPMALFORMED") \
        .load(os.path.join(mimic3_path, 'DIAGNOSES_ICD.csv'))
    diagnoses = diagnoses.join(codes, on=['ICD9_CODE'])
    diagnoses = diagnoses.withColumn('SUBJECT_ID', diagnoses.SUBJECT_ID.cast('int'))
    diagnoses = diagnoses.withColumn('HADM_ID', diagnoses.HADM_ID.cast('int'))
    diagnoses = diagnoses.withColumn('SEQ_NUM', diagnoses.SEQ_NUM.cast('int'))
    return diagnoses


def read_events_table_by_row(mimic3_path, table):
    nb_rows = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    reader = csv.DictReader(open(os.path.join(mimic3_path, table.upper() + '.csv'), 'r'))
    for i, row in enumerate(reader):
        if 'ICUSTAY_ID' not in row:
            row['ICUSTAY_ID'] = ''
        yield row, i, nb_rows[table.lower()]


def count_icd_codes(diagnoses, output_path=None):
    codes = diagnoses[['ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE']].drop_duplicates().set_index('ICD9_CODE')
    codes['COUNT'] = diagnoses.groupby('ICD9_CODE')['ICUSTAY_ID'].count()
    codes.COUNT = codes.COUNT.fillna(0).astype(int)
    codes = codes[codes.COUNT > 0]
    # if output_path:
    #     codes.to_csv(output_path, index_label='ICD9_CODE')
    return codes.sort_values('COUNT', ascending=False).reset_index()

def count_icd_codes_df(diagnoses_df, output_path=None):
    codes = diagnoses_df.select('ICD9_CODE', 'SHORT_TITLE', 'LONG_TITLE').distinct()
    codes = diagnoses_df.groupby('ICD9_CODE').agg(psql.count('ICD9_CODE').alias('COUNT')).na.fill(0)
    # codes.COUNT = codes.COUNT.fillna(0).astype(int)
    # codes = codes[codes.COUNT > 0]
    return codes.filter("COUNT > 0").sort('COUNT')
    # if output_path:
    #     codes.to_csv(output_path, index_label='ICD9_CODE')
    # return codes.sort_values('COUNT', ascending=False).reset_index()

def remove_icustays_with_transfers(stays):
    stays = stays[(stays.FIRST_WARDID == stays.LAST_WARDID) & (stays.FIRST_CAREUNIT == stays.LAST_CAREUNIT)]
    return stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'LAST_CAREUNIT', 'DBSOURCE', 'INTIME', 'OUTTIME', 'LOS']]


def merge_on_subject(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID'], right_on=['SUBJECT_ID'])


def merge_on_subject_admission(table1, table2):
    return table1.merge(table2, how='inner', left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def add_age_to_icustays(stays):
    stays["INDATE"] = pd.to_datetime(stays["INTIME"].dt.date)
    stays["DOBDATE"] = pd.to_datetime(stays["DOB"].dt.date)
    # print(stays.loc[stays["DOBDATE"] > stays["INDATE"]])
    stays['AGE'] = (stays["INDATE"].subtract(stays["DOBDATE"])).dt.days // 365.0
    # stays['AGE'] = (stays.INTIME - stays.DOB).apply(lambda s: s / np.timedelta64(1, 's')) / 60.0/60.0/24.0/365.0
    # temp = stays.loc[stays.AGE < 0, ['AGE', 'INDATE', 'DOBDATE']]
    # print(temp)
    # temp2 = stays.loc[stays.AGE > 0, ['AGE', 'INDATE', 'DOBDATE']]
    # print(temp2)
    stays.loc[stays.AGE < 0, 'AGE'] = 90
    return stays


def add_inhospital_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.ADMITTIME <= stays.DOD) & (stays.DISCHTIME >= stays.DOD))
    # print(mortality[mortality == True])
    mortality = mortality | (stays.DEATHTIME.notnull() & ((stays.ADMITTIME <= stays.DEATHTIME) & (stays.DISCHTIME >= stays.DEATHTIME)))
    # print(mortality[mortality == True])
    stays['MORTALITY'] = mortality.astype(int)
    stays['MORTALITY_INHOSPITAL'] = stays['MORTALITY']
    return stays


def add_inunit_mortality_to_icustays(stays):
    mortality = stays.DOD.notnull() & ((stays.INTIME <= stays.DOD) & (stays.OUTTIME >= stays.DOD))
    mortality = mortality | (
                stays.DEATHTIME.notnull() & ((stays.INTIME <= stays.DEATHTIME) & (stays.OUTTIME >= stays.DEATHTIME)))
    stays['MORTALITY_INUNIT'] = mortality.astype(int)
    return stays


def filter_admissions_on_nb_icustays(stays, min_nb_stays=1, max_nb_stays=1):
    to_keep = stays.groupby('HADM_ID').count()[['ICUSTAY_ID']].reset_index()
    to_keep = to_keep[(to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)][['HADM_ID']]
    stays = stays.merge(to_keep, how='inner', left_on='HADM_ID', right_on='HADM_ID')
    return stays


def filter_icustays_on_age(stays, min_age=18, max_age=np.inf):
    stays = stays[(stays.AGE >= min_age) & (stays.AGE <= max_age)]
    return stays


def filter_diagnoses_on_stays(diagnoses, stays):
    return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
                           left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])

def filter_diagnoses_on_stays_df(diagnoses_df, stays_df):
    return diagnoses_df.join(stays_df.select('SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID'), on=['SUBJECT_ID', 'HADM_ID']).distinct()
    # return diagnoses.merge(stays[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].drop_duplicates(), how='inner',
    #                        left_on=['SUBJECT_ID', 'HADM_ID'], right_on=['SUBJECT_ID', 'HADM_ID'])


def break_up_stays_by_subject(stays, output_path, subjects=None):
    subjects = stays.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up stays by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        stays[stays.SUBJECT_ID == subject_id].sort_values(by='INTIME').to_csv(os.path.join(dn, 'stays.csv'),
                                                                              index=False)
def break_up_stays_by_subject_df(stays, output_path, subjects=None):
    newcols = [c + '_r' for c in subjects.columns]
    subjects2 = subjects.toDF(*newcols)
    stays = stays.join(subjects2.drop_duplicates(['SUBJECT_ID_r']), stays.SUBJECT_ID == subjects2.SUBJECT_ID_r).orderBy(['SUBJECT_ID', 'INTIME'])
    return stays

def break_up_diagnoses_by_subject(diagnoses, output_path, subjects=None):
    subjects = diagnoses.SUBJECT_ID.unique() if subjects is None else subjects
    nb_subjects = subjects.shape[0]
    for subject_id in tqdm(subjects, total=nb_subjects, desc='Breaking up diagnoses by subjects'):
        dn = os.path.join(output_path, str(subject_id))
        try:
            os.makedirs(dn)
        except:
            pass

        diagnoses[diagnoses.SUBJECT_ID == subject_id].sort_values(by=['ICUSTAY_ID', 'SEQ_NUM'])\
                                                     .to_csv(os.path.join(dn, 'diagnoses.csv'), index=False)

def break_up_diagnoses_by_subject_df(diagnoses, output_path, subjects=None):
        newcols = [c + '_r' for c in subjects.columns]
        subjects2 = subjects.toDF(*newcols)
        diagnoses = diagnoses.join(subjects2.drop_duplicates(['SUBJECT_ID_r']),
                           diagnoses.SUBJECT_ID == subjects2.SUBJECT_ID_r).orderBy(['ICUSTAY_ID', 'SEQ_NUM'])
        return diagnoses


def read_events_table_and_break_up_by_subject(mimic3_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    class DataStats(object):
        def __init__(self):
            self.curr_subject_id = ''
            self.curr_obs = []

    data_stats = DataStats()

    def write_current_observations():
        dn = os.path.join(output_path, str(data_stats.curr_subject_id))
        try:
            os.makedirs(dn)
        except:
            pass
        fn = os.path.join(dn, 'events.csv')
        if not os.path.exists(fn) or not os.path.isfile(fn):
            f = open(fn, 'w')
            f.write(','.join(obs_header) + '\n')
            f.close()
        w = csv.DictWriter(open(fn, 'a'), fieldnames=obs_header, quoting=csv.QUOTE_MINIMAL)
        w.writerows(data_stats.curr_obs)
        data_stats.curr_obs = []

    nb_rows_dict = {'chartevents': 330712484, 'labevents': 27854056, 'outputevents': 4349219}
    nb_rows = nb_rows_dict[table.lower()]

    for row, row_no, _ in tqdm(read_events_table_by_row(mimic3_path, table), total=nb_rows,
                                                        desc='Processing {} table'.format(table)):

        if (subjects_to_keep is not None) and (row['SUBJECT_ID'] not in subjects_to_keep):
            continue
        if (items_to_keep is not None) and (row['ITEMID'] not in items_to_keep):
            continue

        row_out = {'SUBJECT_ID': row['SUBJECT_ID'],
                   'HADM_ID': row['HADM_ID'],
                   'ICUSTAY_ID': '' if 'ICUSTAY_ID' not in row else row['ICUSTAY_ID'],
                   'CHARTTIME': row['CHARTTIME'],
                   'ITEMID': row['ITEMID'],
                   'VALUE': row['VALUE'],
                   'VALUEUOM': row['VALUEUOM']}
        if data_stats.curr_subject_id != '' and data_stats.curr_subject_id != row['SUBJECT_ID']:
            write_current_observations()
        data_stats.curr_obs.append(row_out)
        data_stats.curr_subject_id = row['SUBJECT_ID']

    if data_stats.curr_subject_id != '':
        write_current_observations()


 def pySpark_read_events_table_and_break_up_by_subject(mimic3_path, table, output_path,
                                              items_to_keep=None, subjects_to_keep=None):
    obs_header = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM']
    if items_to_keep is not None:
        items_to_keep = set([str(s) for s in items_to_keep])
    if subjects_to_keep is not None:
        subjects_to_keep = set([str(s) for s in subjects_to_keep])

    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    spark = SparkSession.builder.master("local[6]") .getOrCreate()
    sc = spark.sparkContext

    df = spark.read.option("header",True).csv(os.path.join(mimic3_path, table.upper() + '.csv'))
    if not 'ICUSTAY_ID' in df.columns:
        df = df.withColumn('ICUSTAY_ID', func.lit(''))
    df = df.select(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'VALUE', 'VALUEUOM'])
    #broadcast_df = sc.broadcast(df)

    items_to_keep_original = items_to_keep
    i = 0

    for subject in subjects_to_keep:
        i += 1

        outPath = os.path.join(output_path, str(subject))
        outFile = os.path.join(outPath, 'events.csv')
        try:
            os.makedirs(outPath)
        except:
            pass
        
        subject_df = df.filter(df.SUBJECT_ID == subject) 
        #df = df.filter(df.SUBJECT_ID != subject)   

        if items_to_keep_original is not None:
            temp_df = subject_df.filter(subject_df.ITEMID.isin(items_to_keep_original))
            #items_to_keep = sorted(set([int(row.ITEMID) for row in subject_df.distinct().collect()]))
            #print("items is ", items_to_keep)
        else:
            temp_df = subject_df
        
        
        temp_df.toPandas().to_csv(outFile, index=False, mode='w+')
        #temp_df.write.csv(outFile)
        
        if i % 1000 == 0:
            print("{} subjects completed".format(i))
