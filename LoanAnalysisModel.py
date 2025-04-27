import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv('loan.csv')
print(data.head())
print('\n\nColumn Names\n\n')
print(data.columns)

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Loan_Status')
plt.title('Loan Status Distribution')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Gender', hue='Loan_Status')
plt.title('Loan Status by Gender')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Education', hue='Loan_Status')
plt.title('Loan Status by Education')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data['ApplicantIncome'], kde=True, bins=30)
plt.title('Applicant Income Distribution')
plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data['LoanAmount'].dropna(), kde=True, bins=30)
plt.title('Loan Amount Distribution')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', data=data)
plt.title('Loan Amount vs Applicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Loan Amount')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=data, x='Property_Area', hue='Loan_Status')
plt.title('Loan Status by Property Area')
plt.show()

encode = LabelEncoder()
data['Loan_Status'] = encode.fit_transform(data['Loan_Status'])

data.dropna(how='any', inplace=True)

train, test = train_test_split(data, test_size=0.2, random_state=0)

train_x = train.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
train_y = train['Loan_Status']

test_x = test.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
test_y = test['Loan_Status']

train_x = pd.get_dummies(train_x)
test_x = pd.get_dummies(test_x)

train_x, test_x = train_x.align(test_x, join='left', axis=1, fill_value=0)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

print('shape of training data : ', train_x.shape)
print('shape of testing data : ', test_x.shape)

model = LogisticRegression(max_iter=2000)  

model.fit(train_x, train_y)

predict = model.predict(test_x)

print('Predicted Values on Test Data', predict)

print('\n\nAccuracy Score on test data : \n\n')
print(accuracy_score(test_y, predict))
