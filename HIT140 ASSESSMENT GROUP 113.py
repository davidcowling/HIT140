import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as stm
import scipy.stats as st
import math
import matplotlib.pyplot as plt
import os
import textwrap

# Print current working directory
print("Current working directory:", os.getcwd())

# Define column names
column_names = ['Subject Identifier', 'Jitter %', 'Jitter Absolute Microseconds', 'Jitter relative amplitude perturbation', 'Jitter 5-point period perturbation quotient', 'Jitter average absolute difference of differences between jitter cycles',
                'Shimmer in %', 'Absolute shimmer in decibels', 'Shimmer 3 point amplitude perturbation quotient', 'Shimmer 5 point amplitude perturbation quotient', 'Shimmer 11 point amplitude perturbation quotient', 'Shimmer as average absolute differences between consecutive differences between the amplitudes of shimmer cycles',
                'Autocorrelation between NHR and HNR', 'Noise to Harmonic Ratio (NHR)', 'Harmonic to Noise Ratio (HNR)', 'Median Pitch', 'Mean Pitch', 'Standard Deviation of Pitch', 'Min Pitch', 'Max Pitch', 'Number of pulses', 'Number of periods',
                'Mean Period', 'Standard Deviation of Period', 'Fraction of unvoiced frames', 'Number of voice breaks', 'Degree of voice breaks', 'UPDRS', 'PD indicator']

# Read data from file
df = pd.read_csv('po1_data.txt', names=column_names)

# Group Dataframe
grouped_df = df.groupby('Subject Identifier').mean()

# Split into 2 arrays
pdarray = grouped_df[grouped_df['PD indicator'] == 1]
nopdarray = grouped_df[grouped_df['PD indicator'] == 0]

# Calculate confidence level and significance level
conf_lvl = 0.99
sig_lvl = 1 - conf_lvl

# Function to display Graphs
def Graphs():
    
    # Fix title and y label overlapping against other graphs
    titleWrap = "\n".join(textwrap.wrap(column_name, width=40))
    
    # CREATE a line graph of the entire dataset
    plt.figure(figsize=(14, 7))
    plt.subplot(1,3,1)
    plt.axhline(y=x_barColumnTotal, color='blue', label=f'Total mean: {x_barColumnTotal:.5f}')
    plt.axhline(y=ci_lowColumnTotal, color='orange', linestyle='dashed', label='95% CI')
    plt.axhline(y=ci_uppColumnTotal, color='orange', linestyle='dashed')
    plt.plot(np.arange(len(ColumnTotal)), ColumnTotal, marker='o', color='black', label=f'Individual Data - Sample size: {nColumnTotal}')
    
    # Add labels and title
    plt.xlabel('Data Point Index')
    
    # Wrap the column name to fit within the graph
    plt.ylabel(titleWrap)
    plt.title(titleWrap + '\n' + ' and Confidence Interval for all data')
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()
    
    
    
    # CREATE a line graph of PD and NONPD overlapping
    plt.subplot(1,3,2)
    
    #Plot graph of PD
    plt.axhline(y=x_barColumnPD, color='blue', label=f'Mean of PLWPD: {x_barColumnPD:.5f}')
    plt.axhline(y=ci_lowColumnPD, color='orange', linestyle='dashed', label='95% CI of PLWPD')
    plt.axhline(y=ci_uppColumnPD, color='orange', linestyle='dashed')
    plt.plot(np.arange(len(ColumnPD)), ColumnPD, marker='o', color='red', label=f'PLWPD - Sample size: {nColumnPD}')
    

    #Plot graph of NON PD
    plt.axhline(y=x_barColumnNOPD, color='magenta', label=f'Mean of Non PD: {x_barColumnNOPD:.5f}')
    plt.axhline(y=ci_lowColumnNOPD, color='red', linestyle='dashed', label='95% CI of Non PD')
    plt.axhline(y=ci_uppColumnNOPD, color='red', linestyle='dashed')
    plt.plot(np.arange(len(ColumnNOPD)), ColumnNOPD, marker='o', color='cyan', label=f'Individual Non PD - Sample size: {nColumnNOPD}')
        
    # Add labels and title
    plt.xlabel('Data Point Index')
    
    # Wrap the column name to fit within the graph
    plt.ylabel(titleWrap)
    plt.title(titleWrap + '\n' + ' and Confidence Interval for PD vs Non PD')
    
    # Add legend and adjust layout
    plt.legend()
    plt.tight_layout()
    
    
    # CREATE a bar graph of PD and NONPD MEAN side by side
    plt.subplot(1,3,3)
    
    # Round to 5 decimal places to avoid overlapping numbers
    plt.bar(['PD Mean: ' + str("%.5f" % x_barColumnPD), 'Non-PD Mean: ' + str("%.5f" % x_barColumnNOPD)], [x_barColumnPD, x_barColumnNOPD], color=['blue', 'magenta'])
    plt.xlabel('Group')
    
    # Wrap the column name to fit within the graph
    plt.ylabel(titleWrap)
    plt.title(titleWrap + '\n' + ' and Confidence Interval for PD vs Non PD')
    
    # Add legend and adjust layout
    plt.tight_layout()
    plt.show()
    return plt.show()

# Calculations and graph display      
for column_name in grouped_df.columns:
    if column_name != "Subject Identifier" and column_name != "PD indicator":
        # Sort arrays before using them
        ColumnTotal = np.sort(grouped_df[column_name].to_numpy())
        ColumnPD = np.sort(pdarray[column_name].to_numpy())
        ColumnNOPD = np.sort(nopdarray[column_name].to_numpy())

        # Calculate the mean
        x_barColumnTotal = np.mean(ColumnTotal)
        x_barColumnPD = np.mean(ColumnPD)
        x_barColumnNOPD = np.mean(ColumnNOPD)

        # Calculate Standard Deviation
        sColumnTotal = np.std(ColumnTotal)
        sColumnPD = np.std(ColumnPD)
        sColumnNOPD = np.std(ColumnNOPD)

        # Calculate Entries
        nColumnTotal = len(ColumnTotal)
        nColumnPD = len(ColumnPD)
        nColumnNOPD = len(ColumnNOPD)

        # Display Mean, Standard Deviation, and Sample Size Totals
        print("\n")
        print(column_name, " Mean of all entries:", x_barColumnTotal)
        print(column_name, " Standard Deviation:", sColumnTotal)
        print(column_name, " Sample Size:", nColumnTotal)
    
        # Display Mean, Standard Deviation and Sample Size for those with Parkinsons Disease
        print("\n")
        print(column_name," Mean of entries with PD of those with Parkinsons Disease:", x_barColumnPD)
        print(column_name," Standard Deviation of those with Parkinsons Disease:", sColumnPD)
        print(column_name," Sample Size of those with Parkinsons Disease:", nColumnPD)
    
        # Display Mean, Standard Deviation and Sample Size for those withoit Parkinsons Disease
        print("\n")
        print(column_name, " Mean of those without Parkinsons Disease:", x_barColumnNOPD)
        print(column_name, " Standard Deviation of those without Parkinsons Disease:", sColumnNOPD)
        print(column_name, " Sample Size of those without Parkinsons Disease:", nColumnNOPD)
    
        # Calculate standard error
        std_errTotal = sColumnTotal / math.sqrt(nColumnTotal)
        std_errPD = sColumnPD / math.sqrt(nColumnPD)
        std_errNOPD = sColumnNOPD / math.sqrt(nColumnNOPD)

        # Calculate degrees of freedom
        degTotal = nColumnTotal - 1
        degPD = nColumnPD - 1
        degNOPD =  nColumnNOPD - 1

        # Calculate confidence interval
        ci_lowColumnTotal, ci_uppColumnTotal = stm._tconfint_generic(x_barColumnTotal, std_errTotal, degTotal, alpha=sig_lvl, alternative='two-sided')
        ci_lowColumnPD, ci_uppColumnPD = stm._tconfint_generic(x_barColumnPD, std_errPD, degPD, alpha=sig_lvl, alternative='two-sided')
        ci_lowColumnNOPD, ci_uppColumnNOPD = stm._tconfint_generic(x_barColumnNOPD, std_errNOPD, degNOPD, alpha=sig_lvl, alternative='two-sided')

        # Display Confidence Intervals
        print("\n")
        print(column_name, "Confidence Interval of ALL entries: %.2f and %.2f" % (ci_lowColumnTotal, ci_uppColumnTotal))
        print(column_name, "Confidence Interval of those with Parkinsons Disease: %.2f and %.2f" % (ci_lowColumnPD, ci_uppColumnPD))
        print(column_name, "Confidence Interval of those withOUT Parkinsons Disease: %.2f and %.2f" % (ci_lowColumnNOPD, ci_uppColumnNOPD))
        
        # Calculate T statistics/Hypothesis Testing
        t_stats, p_val = st.ttest_ind_from_stats(x_barColumnPD, sColumnPD, nColumnPD, x_barColumnNOPD, sColumnNOPD, nColumnNOPD, equal_var=False, alternative='two-sided')
        print("\t t-statistic (t*): %.2f" % t_stats)
        print("\t p-value: %.4f" % p_val)

        print("\n Conclusion:")
        if p_val < 0.05:
            print("\t We reject the null hypothesis. This suggests that the observed difference between the two groups is unlikely to have occurred due to random chance alone.")
        else:
            print("\t We accept the null hypothesis. This suggests that we have not found significant differences")
        
        #Call the functions that displays the graphs
        Graphs()