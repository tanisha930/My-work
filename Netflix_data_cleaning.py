import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("C:/Users/tasfi/OneDrive/Documents/GitHub/my_work/vs_code/netflix_titles.csv")

##print(df.head())
print("Original shape:", df.shape)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle missing values
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Not Specified', inplace=True)
df['country'].fillna('Not Specified', inplace=True)
df['date_added'] = pd.to_datetime(df['date_added'])
df['rating'].fillna('Unknown', inplace=True)
df['duration'].fillna('Unknown', inplace=True)

# Strip whitespaces from columns
df.columns = df.columns.str.strip()

# Save cleaned file
df.to_csv('netflix_titles_cleaned.csv', index=False)

print("Cleaned shape:", df.shape)

sns.countplot(data=df, x='type', palette='Set2')
plt.title("Content Type Distribution")
plt.xlabel("Type")
plt.ylabel("Count")
plt.savefig("type_distribution.png")
plt.show()

top_countries = df['country'].value_counts().head(10)
top_countries.plot(kind='barh', color='tomato')
plt.title("Top 10 Content Producing Countries")
plt.xlabel("Number of Titles")
plt.savefig("")
plt.show()
