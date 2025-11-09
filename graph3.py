
import pandas as pd
import matplotlib.pyplot as plt

# File path
file_path = "C:\Avishek\Student Arbeit\Experiment Day 01_Data-.xlsx"

# Read the Excel file
df = pd.read_excel(file_path, header=None)  # No header assumed

# Extract the required data
x = df.iloc[2:532, 1]  # 2nd column (Time), rows 3 to 532
y = df.iloc[2:532, 12]  # 13th column (Roll gap), rows 3 to 532

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.xlabel("Time")
plt.ylabel("Average rolling temperature")
plt.title("Average rolling temperature vs Time")
plt.grid()
plt.show()
