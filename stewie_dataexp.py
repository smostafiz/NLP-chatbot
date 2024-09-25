import pandas as pd
import matplotlib.pyplot as plt

"""Read the CSV file into a DataFrame."""
df = pd.read_csv('twcs.csv')

"""Print Basic Dataset information."""
print("Basic Dataset Information:")
print(df.info())

"""Print summary statistics."""
print("\nSummary Statistics:")
print(df.describe())

"""Create a bar graph for the number of unique authors."""
plt.figure(figsize=(10, 6))
author_counts = df['author_id'].value_counts()
author_counts[:10].plot(kind='bar')
plt.title("Top 10 Authors by Tweet Count")
plt.xlabel("Author")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

"""Create a bar graph for the number of unique companies."""
plt.figure(figsize=(10, 6))
company_counts = df['response_tweet_id'].value_counts()
company_counts[:10].plot(kind='bar')
plt.title("Top 10 Companies by Tweet Count")
plt.xlabel("Company")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
