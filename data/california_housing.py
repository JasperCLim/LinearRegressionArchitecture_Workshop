from sklearn.datasets import fetch_california_housing

# Fetch the dataset and return it as a pandas DataFrame
housing = fetch_california_housing(as_frame=True)

# Separate features (X) and target/labels (y)
X = housing.frame.drop(columns=['MedHouseVal'])
y = housing.target

# Inspect the first few rows
print(X.head())
housing.frame.to_csv('cleaned_data.csv', index=False)