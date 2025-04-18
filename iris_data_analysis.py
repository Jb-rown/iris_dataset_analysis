#import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visualization
sns.set_style("whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("✅ Dataset loaded successfully.")

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    
    # Check for missing values
    print("\nMissing Values:\n", df.isnull().sum())

    # Data types
    print("\nData types:\n", df.dtypes)

except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"An error occurred: {str(e)}")

# Task 2: Basic Data Analysis
# Basic statistics
print("\nBasic Statistics:\n", df.describe())


# Group by species and compute mean for each numerical column
print("\nMean values by Species:")
species_means = df.groupby('species').mean()
print(species_means)

# Observations
print("\nObservations:")
print("1. Setosa species tends to have smaller measurements across all features.")
print("2. Virginica species generally has the largest measurements.")
print("3. Versicolor falls between Setosa and Virginica in most measurements.")

# Task 3: Data Visualization
# Create a figure with 2x2 subplots
plt.figure(figsize=(15, 10))

# 1. Bar Chart: Average sepal length by species
plt.subplot(2, 2, 1)
sns.barplot(x='species', y='sepal length (cm)', data=df)
plt.title('Average Sepal Length by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')

# 2. Histogram: Distribution of petal length
plt.subplot(2, 2, 2)
plt.hist(df['petal length (cm)'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')

# 3. Scatter Plot: Sepal length vs Petal length
plt.subplot(2, 2, 3)
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
               species_data['petal length (cm)'], 
               label=species, 
               alpha=0.6)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()

# 4. Line Chart: Mean measurements across species
plt.subplot(2, 2, 4)
for column in df.columns[:-1]:  # Exclude species column
    plt.plot(species_means.index, species_means[column], marker='o', label=column)
plt.title('Mean Measurements by Species')
plt.xlabel('Species')
plt.ylabel('Measurement (cm)')
plt.legend()

# # Line chart (fake trend using index)
# plt.figure(figsize=(8, 4))
# plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length")
# plt.title("Sepal Length Over Observations")
# plt.xlabel("Index")
# plt.ylabel("Sepal Length (cm)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Adjust layout and save
plt.tight_layout()
plt.savefig('iris_visualizations.png')
plt.close()

