import pandas as pd

# no truncation when printing df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

train_label_df = pd.read_csv('../train_listfile_by.csv')
df = pd.DataFrame()

row_1 = train_label_df.iloc[0]
df = pd.read_csv('../train/' + row_1['stay'])

group = ""
for i in row_1['stay']:
    if i < '0' or i > '9':
        break
    group += i

df['time_idx'] = df.index
df['group'] = int(group)
df['mortality'] = row_1['y_true']

# print(df)

for index, row in train_label_df.iterrows():
    # print(row['stay'], row['y_true'])
    if index == 0:
        continue
    temp_df = pd.read_csv('../train/' + row['stay'])
    temp_df['time_idx'] = temp_df.index

    group = ""
    for i in row['stay']:
        if i < '0' or i > '9':
            break
        group += i
    temp_df['group'] = int(group)

    temp_df['mortality'] = row['y_true']
    df = pd.concat([df, temp_df], ignore_index=True, axis=0)

print(df)
df.to_csv('train_timeseries.csv')
