import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load local dataset
# -------------------------------
# Make sure "german.data" is in the same folder as your script
df = pd.read_csv("german.data", header=None, sep=r"\s+")

# Column indices based on UCI dataset documentation
# Numeric columns you want: 
# Attribute2 = Duration
# Attribute5 = Credit Amount
# Attribute8 = Installment rate
# Attribute11 = Residence since
# Attribute13 = Age
# Attribute16 = Number of existing credits
# Attribute18 = Number of people being liable

X = df.iloc[:, :-1]  # all columns except last
y = df.iloc[:, -1]   # last column = target

# Extract numeric columns
credit_amount = X[4]        # Attribute5
duration = X[1]             # Attribute2
installment_rate = X[7]     # Attribute8
residence_since = X[10]     # Attribute11
age = X[12]                 # Attribute13
number_of_existing_credits = X[15]  # Attribute16
num_people_being_liable = X[17]     # Attribute18

# Convert y to binary: 1 = good, 0 = bad
y_binary = (y == 1).astype(int)

# -------------------------------
# Plotting
# -------------------------------
def boxplot_feature(feature, feature_name):
    sns.boxplot(x=y_binary, y=feature)
    plt.xlabel('Credit Risk (1=Good, 0=Bad)')
    plt.ylabel(feature_name)
    plt.show()

boxplot_feature(credit_amount, "Credit Amount")
boxplot_feature(duration, "Duration (months)")
boxplot_feature(age, "Age")
boxplot_feature(installment_rate, "Installment Rate (%)")
boxplot_feature(number_of_existing_credits, "Number of Existing Credits")
boxplot_feature(num_people_being_liable, "People Being Liable")
boxplot_feature(residence_since, "Residence Since (years)")
