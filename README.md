# 🌸 Iris Dataset Analysis with Pandas & Matplotlib

This project demonstrates basic data analysis and visualization using the classic **Iris dataset**.  
The dataset is loaded using `sklearn`, explored with `pandas`, and visualized using `matplotlib` and `seaborn`.

---

## 📌 Project Overview

- ✅ Load and explore the Iris dataset
- 📈 Perform basic statistics and group analysis
- 📊 Visualize insights with 4 types of charts:
  - Line chart
  - Bar chart
  - Histogram
  - Scatter plot


## 📁 Project Structure
    iris-analysis/

            ├── iris_data_analysis.py # Jupyter Notebook for analysis
            ├── README.md # Project overview and instructions



## 📦 Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

Install dependencies with:

    pip install pandas matplotlib seaborn scikit-learn notebook

## How to Run
1. Clone this repository or download the files.

2. Open a terminal in the project folder.

3. Start Jupyter Notebook:


        jupyter notebook

4. Open iris_data_analysis.ipynb and run the cells step-by-step.

## Visualizations Included
- Line chart of Sepal Length over index

- Bar chart of average Petal Length per species

- Histogram of Sepal Width distribution

- Scatter plot of Sepal Length vs Petal Length with species hue

## Insights
- The analysis includes:

- Dataset exploration

- Summary statistics

- Group-based averages

- Customized plots with titles, labels, and legends

 ## Dataset Source
The Iris dataset is loaded using:

````python
 from sklearn.datasets import load_iris

