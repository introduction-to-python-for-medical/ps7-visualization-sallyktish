import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer 
data = load_breast_cancer()
import pandas as pd
print(data.DESCR)
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target # Adding target variable to the dataframe
df.sample(5)
df.describe()
df.dtypes
features = list(df.columns)
print("Available features:", features)
selected_features = [features[0], features[4], features[5], features[6], features[7]]
print("Selected features: ", selected_features)
fig, axs  = plt.subplots(1, len(selected_features), figsize = (20,3))

for ax, f in zip(axs, selected_features):
    ax.hist(df[f], bins=5, color='skyblue', edgecolor='black')
    ax.set_xlabel(f)
  reference_feature = selected_features[1]
y = df[reference_feature]

fig, axs  = plt.subplots(1, len(selected_features), figsize = (20,3))

for ax, f in zip(axs, features):
  ax.scatter(df[f], y)
  ax.set_xlabel(f)

plt.show()
reference_feature = selected_features[0]  # The reference feature
comparison_feature = selected_features[1]  # A feature to compare to

# Create a scatter plot for the selected pair
plt.figure(figsize=(8, 6))
plt.scatter(df[reference_feature], df[comparison_feature], alpha=0.6)
plt.xlabel(reference_feature)
plt.ylabel(comparison_feature)

# Save the plot as an image file
plt.savefig('correlation_plot.png')

plt.show()
