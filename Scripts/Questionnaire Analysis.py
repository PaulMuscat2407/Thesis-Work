import pandas as pd
from scipy.stats import wilcoxon

# Load the Excel file
file_path = 'GPT Response Preference.xlsx'
sheet_name = 'Analysis'
df = pd.read_excel(file_path, sheet_name=sheet_name)

start_row = 0
end_row = 62

# Change column name to the columns you wish to compare against each other
data1 = pd.to_numeric(df['medium3'].iloc[start_row:end_row + 1])
data2 = pd.to_numeric(df['long3'].iloc[start_row:end_row + 1])


# Perform the Signed Rank test
stat, p = wilcoxon(data1, data2,alternative='two-sided')

print(f'Signed Rank Test statistic: {stat}')
print(f'P-value: {p}')

# Interpretation based on p-value
alpha = 0.05
if p < alpha /2 or p > 1 - (alpha / 2):
    print('Different distribution (reject H0)')
else:
    print('Same distribution (fail to reject H0)')
    
# Neyman Pearson methodology
# There is sufficient statistical evudebce at tbe alpha = 0.05 level of significance to reject H0 in favour Ha.
# In other words, there is only a 0.05 probability of a type 1 error (we reject h0 when h0 is true)
# We believe that the median of both sets are NOT the same (in this case, the median of the first set is larger)

# There is insufficient statistical evidence at the alpha = 0.05 level of significance to reject H0. (H0 is accepted by default, typically)