import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Use a raw string literal for the file path to avoid backslash issues in Windows
file_path = r'C:/Users/i/Desktop/python/datasets/test.csv'

# Load the Titanic dataset
df = pd.read_csv(file_path)

# Display the first few rows to understand the structure
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Handle missing values

# Replace missing 'Age' values with median age
median_age = df['Age'].median()
df['Age'].fillna(median_age, inplace=True)

# Replace missing 'Embarked' values with the mode
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# Drop 'Cabin' column due to high percentage of missing values
df.drop('Cabin', axis=1, inplace=True)

# Verify no more missing values
print(df.isnull().sum())

# Exploratory Data Analysis (EDA)

# Summary statistics
print(df.describe())

# Visualizations

# Example: Survival by gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Example: Age distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Example: Survival by passenger class
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Example: Fare distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Fare'], bins=30, kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Count')
plt.show()

# Example: Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()
