# CE 49X - Lab 2: Soil Test Data Analysis

# Student Name: Özgür Cemal_Özer_____
# Student ID: ____2021403024
# Date: __16/10/2025_______

import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load the soil test dataset from a CSV file.
    
    Parameters:
        file_path (str): The path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded DataFrame, or None if the file is not found.
    """
    # TODO: Implement data loading with error handling
    pass
    try:
        df = pd.read_csv(file_path)
        return df

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None

    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None


    
def clean_data(df):
    """
    Clean the dataset by handling missing values and removing outliers from 'soil_ph'.
    
    For each column in ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']:
    - Missing values are filled with the column mean.
    
    Additionally, remove outliers in 'soil_ph' that are more than 3 standard deviations from the mean.
    
    Parameters:
        df (pd.DataFrame): The raw DataFrame.
        
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df_cleaned = df.copy()
    cols = ['soil_ph', 'nitrogen', 'phosphorus', 'moisture']
    
    # TODO: Fill missing values in each specified column with the column mean
    for col in cols:
        mean_val = df_cleaned[col].mean()
        df_cleaned[col] = df_cleaned[col].fillna(mean_val)
    
    # TODO: Remove outliers in 'soil_ph': values more than 3 standard deviations from the mean
    mean_ph = df_cleaned['soil_ph'].mean()
    std_ph = df_cleaned['soil_ph'].std()
    df_cleaned = df_cleaned[
        (df_cleaned['soil_ph'] > mean_ph - 3 * std_ph) &
        (df_cleaned['soil_ph'] < mean_ph + 3 * std_ph)
        ]
    
    print(df_cleaned.head())
    return df_cleaned
def compute_statistics(df, column):
    """
    Compute and print descriptive statistics for the specified column.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column for which to compute statistics.
    """
    # TODO: Calculate minimum value
    min_val = df[column].min()
    
    # TODO: Calculate maximum value
    max_val = df[column].max()
    
    # TODO: Calculate mean value
    mean_val = df[column].mean()
    
    # TODO: Calculate median value
    median_val = df[column].median()
    
    # TODO: Calculate standard deviation
    std_val = df[column].std()
    
    print(f"\nDescriptive statistics for '{column}':")
    print(f"  Minimum: {min_val}")
    print(f"  Maximum: {max_val}")
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Median: {median_val:.2f}")
    print(f"  Standard Deviation: {std_val:.2f}")

def main():
    # TODO: Update the file path to point to your soil_test.csv file
    file_path = 'soil_test.csv'  # Update this path as needed

    
    # TODO: Load the dataset using the load_data function
    df = load_data(file_path)
    
    # TODO: Clean the dataset using the clean_data function
    if df is not None:
        df_clean = clean_data(df)

    
    # TODO: Compute and display statistics for the 'soil_ph' column
    compute_statistics(df_clean, 'soil_ph')

    # TODO: (Optional) Compute statistics for other columns
    # compute_statistics(df_clean, 'nitrogen')
    compute_statistics(df_clean, 'nitrogen')
    # compute_statistics(df_clean, 'phosphorus')
    compute_statistics(df_clean, 'phosphorus')
    # compute_statistics(df_clean, 'moisture')
    compute_statistics(df_clean, 'moisture')



if __name__ == '__main__':
    main()

# =============================================================================
# REFLECTION QUESTIONS
# =============================================================================
# Answer these questions in comments below:

# 1. What was the most challenging part of this lab?
# Answer: I found it a bit difficult to understand how Python decides when to run
# the main() function automatically and why that structure is used. 

# 2. How could soil data analysis help civil engineers in real projects?
# Answer: Soil data analysis helps civil engineers assess the suitability of
# soil for construction projects. By analyzing properties like soil pH, moisture,
# and nutrient content, engineers can make better decisions about foundation design,
# material selection, and site safety. It also helps identify potential soil issues
# early in the planning stage.



# 3. What additional features would make this soil analysis tool more useful?
# Answer: Visualizing the variation in the properties of soil samples taken from
# the same location under different weather conditions and at various times.


# 4. How did error handling improve the robustness of your code?
# Answer: Error handling made the code more reliable and user-friendly. 
# For example, by catching the FileNotFoundError, the program can display a clear
# message instead of crashing. It ensures that unexpected issues, like
# missing files or invalid inputs, do not stop the entire program from running.