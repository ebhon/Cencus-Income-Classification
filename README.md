# Income Prediction Analysis Project

## Overview

This project analyzes a dataset containing various demographic, educational, and employment-related features to predict individuals' income levels. The primary objective is to understand the factors influencing income disparities and to build a machine learning model to predict income categories based on these features. The analysis will inform policies that can help reduce income inequality, improve access to education, and promote fairer employment opportunities.

## Dataset

The dataset consists of 15 columns and 48,842 rows of data. It includes both numerical and categorical features related to personal attributes, employment details, and income.

### Key Columns:
- **Numerical Features**: 
  - age
  - fnlwgt
  - educational-num
  - capital-gain
  - capital-loss
  - hours-per-week

- **Ordinal Features**: 
  - workclass
  - education
  - marital-status
  - occupation
  - relationship
  - race
  - gender
  - income

- **Categorical Features**:
  - native-country

Dataset source: [Adult Income Dataset on Kaggle](https://www.kaggle.com/datasets/wenruliu/adult-income-dataset)

## Objectives

The primary objectives of the project are:
- **Exploratory Data Analysis (EDA)**: Understand the distribution of key features, identify trends, detect anomalies, and handle missing values.
- **Feature Engineering**: Clean and preprocess data, including handling missing values, encoding categorical variables, and scaling numerical features.
- **Modeling**: Train machine learning models to predict income levels (greater than or less than 50 million IDR).
- **Model Evaluation**: Evaluate the performance of different models using cross-validation and metrics like F1-Score, precision, and recall.

## Methodology

1. **Data Cleaning**:
   - Handling missing values in critical columns (`workclass`, `occupation`, `native-country`).
   - Removing duplicate records to ensure accurate analysis.
   
2. **Exploratory Data Analysis (EDA)**:
   - Distribution analysis of key features like age, education, and capital gains.
   - Identifying correlations between variables such as income, education, and occupation.
   - Detecting outliers, especially in numerical features like `capital-gain` and `hours-per-week`.

3. **Feature Engineering**:
   - Handling categorical features by using OneHotEncoder (for features like `gender`, `job-type`, `marital-status`) and TargetEncoder (for features with high cardinality like `education`, `native-country`).
   - Scaling numerical features using standard techniques (e.g., Min-Max Scaling).

4. **Modeling**:
   - Multiple machine learning models were tested, including Random Forest, AdaBoost, and others.
   - Hyperparameter tuning was performed to optimize model performance.
   - Evaluation metrics like F1-Score and cross-validation scores were used to assess the models.

5. **Model Evaluation**:
   - The AdaBoost model was identified as the best-performing model based on F1-Score and consistency across cross-validation folds.

## Insights

- **Income Distribution**: There is a significant income imbalance in the dataset, with most individuals earning less than or equal to 50 million IDR.
- **Education & Income**: Educational attainment plays a significant role in income, with higher education levels generally correlating with higher income.
- **Occupation & Income**: Occupation type influences income levels, with higher-income earners typically in specific sectors.
- **Age & Income**: Older individuals tend to have higher income, reflecting career progression and seniority.
- **Gender and Marital Status**: There are notable correlations between gender, marital status, and income, suggesting that family dynamics may play a role in income levels.

## Recommendations

- **Improve Education Accessibility**: Policies aimed at increasing access to higher education and vocational training programs are key to reducing income inequality.
- **Gender Pay Gap**: Further investigation into the gender pay gap is needed to ensure equal pay for equal work. Promoting skill training and job opportunities for women may help reduce the gap.
- **Targeted Income Support**: Income support policies could be tailored to different family structures, with a particular focus on married couples or dual-income households.

## Model Evaluation

- **AdaBoost Model**: The AdaBoost model achieved the highest F1-Score and demonstrated strong performance in cross-validation. It is the most reliable model for predicting income.
  
### Pros:
- High F1-Score, indicating good balance between precision and recall.
- Hyperparameter tuning improved the model’s stability, reducing variability.
- Consistent performance across cross-validation folds.

### Cons:
- Longer prediction times due to resource-intensive nature.
- May not be ideal for real-time applications in large-scale environments.

## Next Steps

- **Implementation**: The model has shown strong performance, and it's ready for deployment in real-world applications.
- **Optimization**: Given the resource-intensive nature of the AdaBoost model, optimization techniques (such as model pruning) should be explored for faster predictions.

## Conclusion

This project highlights the importance of education, occupation, and family structure in determining income levels. By focusing on these areas, policies can be shaped to address income inequality and improve opportunities for underrepresented groups in the job market.

## Requirements

- Python 3.x
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

To run this project locally, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```
## Deployment
The model has been deployed on Hugging Face Spaces for live testing and inference. You can access the deployed model here: [Income Analysis on Hugging Face](https://huggingface.co/spaces/ebhon/Income-analysis)

## License
This project is licensed under the MIT License - see the LICENSE file for details.