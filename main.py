# Data Visualization with Matplotlib and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('data.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# Basic Data Exploration
print("\nBasic Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# Data Cleaning: Convert columns to appropriate types
# Convert 'Survived' to categorical
df['Survived'] = df['Survived'].astype('category')

# Fill missing values in 'Age' with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Check for missing values in 'Embarked' and fill if necessary
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Visualizations

# 1. Bar Chart: Count of Passengers by Class
plt.figure(figsize=(8, 5))
sns.countplot(x='Pclass', data=df)
plt.title('Count of Passengers by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()

# 2. Bar Chart: Survival Rate by Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='Sex', y='Survived', data=df, errorbar=None)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()

# 3. Scatter Plot: Age vs. Fare
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='coolwarm')
plt.title('Age vs. Fare with Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend(title='Survived', loc='upper right', labels=['No', 'Yes'])
plt.show()

# 4. Heatmap: Correlation Matrix
plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include='number')  # Select only numeric columns
correlation = numeric_cols.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Insights
print("Insights:")
print("1. The bar chart shows that the majority of passengers were in third class.")
print("2. The survival rate was significantly higher for females than for males.")
print("3. The scatter plot reveals a wider distribution of fares paid by younger passengers.")
print("4. The heatmap indicates that 'Fare' and 'Pclass' have a strong correlation, suggesting that higher class passengers tended to pay more.")
