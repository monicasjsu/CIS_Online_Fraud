import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from db.mysql import Engine

# Load data
train_identity = pd.read_csv('../dataset/train_identity.csv')
train_transaction = pd.read_csv('../dataset/train_transaction.csv')
test_identity = pd.read_csv('../dataset/test_identity.csv')
test_transaction = pd.read_csv('../dataset/test_transaction.csv')

print(train_identity.head())
print(train_transaction.head())

# As we can see, that transactionId exists in both data sets. So let's combine the data and identity and transactions
# dataset by joining this TransactionId column
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# Now lets check the total number of rows and columns of the finalized dataset.
print(train.shape)
print(test.shape)

print(train.head())
print(train.describe())

# Lets Check the class values distribution
sns.countplot(x='isFraud', data=train)
plt.show()
train_fraud = train['isFraud']
print(train['isFraud'].value_counts(normalize=True))

# delete unused data frames to reduce memory usage
del train_identity, train_transaction, test_identity, test_transaction

# Lets collect the categorical columns
categorical_columns = train.select_dtypes(include=['object']).columns
print(categorical_columns)
print('Length of Cat columns', len(categorical_columns))

# Lets check the percentage of missing values in all columns
train_null = train.isnull().sum() / len(train) * 100
print('Percentage of nan values in training data\n', train_null)
test_null = test.isnull().sum() / len(train) * 100
print('Percentage of nan values in test data\n', test_null)

# Drop the columns with missing values more than 90%
null_cols_train = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
null_cols_test = [col for col in test.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
# We should only drop the intersection of the null_cols_train and null_cols_test.
cols_to_drop = set(null_cols_train).intersection(null_cols_test)

# Lets verify if we are removing the columns from both data sets
print('Null cols in Train data: ', len(null_cols_train))
print('Null cols in Test data: ', len(null_cols_test))
print('Final Null cols that get dropped: ', len(cols_to_drop))
print('Cols to drop:\n', cols_to_drop)

train.drop(cols_to_drop, axis=1, inplace=True)
test.drop(cols_to_drop, axis=1, inplace=True)

# Lets also check if there are any cols with same values repeated more than 90%
top_value_cols_train = [col for col in train.columns
                            if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
top_value_cols_test = [col for col in test.columns
                           if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

# We should only drop the intersection of the null_cols_train and null_cols_test.
value_cols_to_drop = set(top_value_cols_train).intersection(top_value_cols_test)

# Lets again verify if we are removing the columns from both data sets
print('Null cols in Train data: ', len(value_cols_to_drop))
print('Null cols in Test data: ', len(top_value_cols_test))
print('Final Null cols that get dropped: ', len(value_cols_to_drop))
print('Cols to drop:\n', cols_to_drop)
train.drop(value_cols_to_drop, axis=1, inplace=True)
test.drop(value_cols_to_drop, axis=1, inplace=True)

# Now lets check the shape of train and test data frames
print('Train Shape: ', train.shape)
print('Test Shape: ', test.shape)

# We were able to reduce the number of cols from 434 after merging to 364 after removing missing value cols and
# single value cols greater than 90%

db_conn = Engine.get_db_conn()
train.to_sql('train_cleaned', db_conn, index=False)
test.to_sql('test_cleaned', db_conn, index=False)
train.to_csv('train_cleaned.csv', index=False)
test.to_csv('test_cleaned.csv', index=False)

print('Data has been successfully saved into database')
