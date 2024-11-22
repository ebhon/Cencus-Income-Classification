import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from phik import phik_matrix  

# Path to dataset
data_path = 'adult.csv'

# Load dataset
@st.cache_data
def load_data():
    if not os.path.isfile(data_path):
        st.error(f"File not found: {data_path}")
        return None
    return pd.read_csv(data_path)

def run_eda():
    # Load data
    data = load_data()

    # Check if data is loaded successfully
    if data is not None:
        # Trim whitespace from column names
        data.columns = data.columns.str.strip()
        
        # Sidebar for chart selection
        st.sidebar.title("EDA Menu")
        menu_options = st.sidebar.radio("Select a chart:",
                                         ("Age Distribution Histogram",
                                          "Average Age by Income Category",
                                          "Count by Work Class and Income",
                                          "Average Capital Gain by Education Level",
                                          "Total Hours Worked by Income Category",
                                          "Count by Marital Status and Income",
                                          "Phik Correlation Matrix"))

        # Histogram of Age distribution
        if menu_options == "Age Distribution Histogram":
            st.subheader("Histogram of Age Distribution")
            if 'age' in data.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(data['age'], bins=30, kde=True)
                plt.title('Distribusi Usia')
                plt.xlabel('Usia')
                plt.ylabel('Frekuensi')
                st.pyplot(plt)
                st.write("**Insight:** This histogram shows the age distribution of individuals in the dataset, indicating how age varies among the population.")
            else:
                st.error("Column 'age' not found in the dataset.")

        # Average Age by Income Category
        if menu_options == "Average Age by Income Category":
            st.subheader("Average Age Based on Income Category")
            if 'income' in data.columns and 'age' in data.columns:
                age_income = data.groupby('income')['age'].mean().reset_index()  # Group age by income
                plt.figure(figsize=(10, 6))
                sns.barplot(data=age_income, x='income', y='age')
                plt.title('Rata-rata Usia berdasarkan Kategori Pendapatan')
                plt.xlabel('Kategori Pendapatan')
                plt.ylabel('Rata-rata Usia')
                st.pyplot(plt)
                st.write("**Insight:** This bar plot displays the average age of individuals based on income categories, showing how age correlates with income.")
            else:
                st.error("Required columns not found in the dataset.")

        # Count by Work Class and Income
        if menu_options == "Count by Work Class and Income":
            st.subheader("Count by Work Class and Income")
            if 'workclass' in data.columns and 'income' in data.columns:
                workclass_income = data.groupby(['workclass', 'income']).size().reset_index(name='count')
                plt.figure(figsize=(12, 6))
                sns.barplot(data=workclass_income, x='workclass', y='count', hue='income')
                plt.title('Jumlah Individu berdasarkan Jenis Pekerjaan dan Pendapatan')
                plt.xticks(rotation=45)
                st.pyplot(plt)
                st.write("**Insight:** This plot illustrates the distribution of individuals by their job types and income levels, highlighting job categories that attract higher income.")
            else:
                st.error("Required columns not found in the dataset.")

        # Average Capital Gain by Education Level
        if menu_options == "Average Capital Gain by Education Level":
            st.subheader("Average Capital Gain Based on Education Level")
            if 'education' in data.columns and 'capital-gain' in data.columns:
                capital_gain_education = data.groupby('education')['capital-gain'].mean().reset_index()
                plt.figure(figsize=(12, 6))
                sns.barplot(data=capital_gain_education, x='education', y='capital-gain')
                plt.title('Rata-rata Keuntungan Modal berdasarkan Tingkat Pendidikan')
                plt.xticks(rotation=45)
                st.pyplot(plt)
                st.write("**Insight:** This bar plot indicates the average capital gain across different education levels, suggesting that higher education is associated with greater financial gains.")
            else:
                st.error("Required columns not found in the dataset.")

        # Total Hours Worked by Income Category
        if menu_options == "Total Hours Worked by Income Category":
            st.subheader("Total Hours Worked Based on Income Category")
            if 'income' in data.columns and 'hours-per-week' in data.columns:
                hours_income = data.groupby('income')['hours-per-week'].sum().reset_index()
                plt.figure(figsize=(8, 5))
                sns.barplot(data=hours_income, x='income', y='hours-per-week')
                plt.title('Total Jam Kerja berdasarkan Kategori Pendapatan')
                plt.xlabel('Kategori Pendapatan')
                plt.ylabel('Total Jam Kerja')
                st.pyplot(plt)
                st.write("**Insight:** This plot shows the total number of hours worked for each income category, indicating the relationship between working hours and income.")
            else:
                st.error("Required columns not found in the dataset.")

        # Count by Marital Status and Income
        if menu_options == "Count by Marital Status and Income":
            st.subheader("Count by Marital Status and Income")
            if 'marital-status' in data.columns and 'income' in data.columns:
                relationship_income = data.groupby(['marital-status', 'income']).size().reset_index(name='count')
                plt.figure(figsize=(12, 6))
                sns.barplot(data=relationship_income, x='marital-status', y='count', hue='income')
                plt.title('Jumlah Individu berdasarkan Status Perkawinan dan Pendapatan')
                plt.xticks(rotation=45)
                st.pyplot(plt)
                st.write("**Insight:** This plot shows the distribution of individuals by marital status and income category, providing insights into how marital status may affect income.")
            else:
                st.error("Required columns not found in the dataset.")

        # Phik Correlation Matrix
        if menu_options == "Phik Correlation Matrix":
            st.subheader("Phik Correlation Matrix")
            # List the required columns
            required_columns = ['income', 'age', 'capital-gain', 'hours-per-week', 'marital-status', 'education', 'workclass']
            if all(col in data.columns for col in required_columns):
                # Calculate the Phik correlation matrix
                phik_corr = data.phik_matrix()
                plt.figure(figsize=(12, 8))
                sns.heatmap(phik_corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
                plt.title('Phik Correlation Matrix (Sampled Data)')
                st.pyplot(plt)
                st.write("**Insight:** The Phik correlation matrix reveals the strength and direction of relationships between variables, helping identify multicollinearity and associations within the dataset.")
            else:
                missing_cols = [col for col in required_columns if col not in data.columns]
                st.error(f"Required columns not found in the dataset: {', '.join(missing_cols)}")
    else:
        st.error("Data not loaded successfully.")
