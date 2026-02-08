import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('egypt_real_estate_listings.csv')

print(df.shape)

for i in df.columns:
    missing_count = df[i].isna().sum()
    if missing_count > 0.02 * df.shape[0]:
        print(f"Column '{i}': {missing_count} missing values")
        df.dropna(subset=[i], inplace=True)
df.drop(columns=['down_payment'])

df.isna().sum()

df.info()

df.describe()

df.head()

X =set()
for i in df['type']:
  X.add(i)
print(X)

# frequency of each type of property
sns.countplot(data=df, x='type')
plt.title('Frequency of Property Types')
plt.xticks(rotation=90)
plt.show()

# Remove commas from the 'price' column and convert to numeric, coercing errors to NaN
df['price_numeric'] = pd.to_numeric(df['price'].astype(str).str.replace(',', ''), errors='coerce')

df['Price_Millions'] = df['price_numeric'] / 1000000

#Basic statistics
print("Basic Statistics by Property Type:")
stats = df.groupby('type')['Price_Millions'].describe()
print(stats)

print("\n" + "="*50)
print("COUNT BY PROPERTY TYPE")
print("="*50)
type_counts = df['type'].value_counts()
print(type_counts)

#Average prices bar chart
plt.figure(figsize=(10, 6))
avg_prices = df.groupby('type')['Price_Millions'].mean().sort_values(ascending=False)

bars = plt.bar(avg_prices.index, avg_prices.values, color='lightcoral', alpha=0.7, edgecolor='darkred')
plt.title('Average Price by Property Type')
plt.ylabel('Average Price (Millions)')
plt.xticks(rotation=45)

for bar, value in zip(bars, avg_prices.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{value:.1f}M', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

#Detailed analysis by property type
print("="*50)
print("DETAILED ANALYSIS BY PROPERTY TYPE")
print("="*50)

# Convert all unique property types to strings before sorting
for prop_type in sorted(df['type'].unique().astype(str)):

    if prop_type == 'nan':
        prop_data = df[df['type'].isna()]['Price_Millions']
    else:
        prop_data = df[df['type'] == prop_type]['Price_Millions']

    print(f"\n{prop_type.upper():<12} ({len(prop_data)} properties):")

    # Check if there is data for the current property type before calculating statistics
    if not prop_data.empty:
        print(f"  Average Price: {prop_data.mean():.2f}M")
        print(f"  Median Price:  {prop_data.median():.2f}M")
        print(f"  Min Price:     {prop_data.min():.2f}M")
        print(f"  Max Price:     {prop_data.max():.2f}M")
        # Ensure min() and max() return scalar values before subtraction
        min_price = prop_data.min()
        max_price = prop_data.max()
        if pd.isna(min_price) or pd.isna(max_price):
          print(f"  Price Range:   N/A")
        else:
          print(f"  Price Range:   {max_price - min_price:.2f}M")
        print(f"  Std Dev:       {prop_data.std():.2f}M")
    else:
        print("  No data available for this property type.")

types = df['type']

# Correctly access the 'size' column and convert it to string type
size_sqm = []
for size in df['size'].astype(str):
    parts = size.split('/')
    if len(parts) > 1:
        sqm_part = parts[1].strip()
        sqm_value = sqm_part.split('sqm')[0].strip().replace(',', '')
        size_sqm.append(pd.to_numeric(sqm_value, errors='coerce'))
    else:
        size_sqm.append(np.nan)

df['Size_sqm'] = size_sqm

print("Data Overview:")
print(df[['type', 'size', 'Size_sqm']].head())


print("\nAVERAGE SIZE BY PROPERTY TYPE:")
avg_sizes = df.groupby('type')['Size_sqm'].mean().round(1)
print(avg_sizes)

plt.figure(figsize=(10, 5))
avg_sizes.plot(kind='bar', color='blue')
plt.title('Average Size by Property Type')
plt.ylabel('Size (sqm)')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

bedroom_numbers = []
bedroom_numbers = []

for bed in df['bedrooms']:
    if pd.isna(bed):
        bedroom_numbers.append(None)
        continue

    bed = str(bed)

    if '+' in bed:
        bed_num = bed.split('+')[0].strip()
    else:
        bed_num = bed.strip()

    bedroom_numbers.append(bed_num)


df['Bedroom_Num'] = bedroom_numbers

print("Bedroom Data:")
print(df[['type', 'bedrooms', 'Bedroom_Num']].head())


plt.figure(figsize=(10, 5))
avg_bedrooms = df.groupby('type')['Bedroom_Num'].apply(lambda x: pd.to_numeric(x.replace('studio', '1').replace('nan', np.nan), errors='coerce').mean()).round(1)
avg_bedrooms.plot(kind='bar', color='green')
plt.title('Average Number of Bedrooms by Property Type')
plt.ylabel('Number of Bedrooms')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

# Average bathrooms by type
df['bathrooms_numeric'] = pd.to_numeric(df['bathrooms'], errors='coerce')
plt.figure(figsize=(10, 5))
avg_bathroom = df.groupby('type')['bathrooms_numeric'].mean()
avg_bathroom.plot(kind='bar', color='orange')
plt.title('Average Number of Bathrooms by Property Type')
plt.ylabel('Number of Bathrooms')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()

cross_tab = pd.crosstab(df['type'], df['payment_method'])
print("CROSS-TABULATION:")
print(cross_tab)



plt.figure(figsize=(10, 5))
cross_tab.plot(kind='bar', stacked=True)
plt.title('Property Type vs payment_method')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.legend(title='Size Group')
plt.show()


plt.figure(figsize=(8, 8))
size_counts = df['payment_method'].value_counts()
plt.pie(size_counts.values, labels=size_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('payment method Distribution')
plt.show()


unique_locations = df['location'].unique()
print(unique_locations)


locations_to_find = ['Gouna', 'New Zayed City', 'Sheikh Zayed City', 'Hurghada', 'North Coast', 'Ras Al Hekma']
for i in locations_to_find:
  print(i,df['location'].str.contains(i).sum())


df['Price_per_sqft'] = df['price_numeric'] / df['Size_sqm']



results = {}
for location in locations_to_find:
    location_properties = df[df['location'].str.contains(location, na=False, case=False)]
    if len(location_properties) > 0:
        avg_price = location_properties['Price_per_sqft'].mean().round(2)
        results[location] = avg_price
    else:
        results[location] = 0

avg_price_per_sqft = pd.Series(results)

print(f"Found properties in {len([x for x in results.values() if x > 0])} out of {len(locations_to_find)} locations")
print("\n" + "=" * 50)
print("AVERAGE PRICE PER SQ FT")
print("=" * 50)
for location, price in avg_price_per_sqft.items():
    count = len(df[df['location'].str.contains(location, na=False, case=False)])
    print(f"📍 {location}: EGP {price:,.2f} ({count} properties)")


plt.figure(figsize=(10, 6))
avg_price_per_sqft.sort_values(ascending=True).plot(kind='barh', color='lightblue')
plt.title('Average Price per Square Foot', fontsize=14, fontweight='bold')
plt.xlabel('Price per Sq Ft (EGP)')
plt.ylabel('Location')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

df.info()

from sklearn.model_selection import train_test_split

t = df['price_numeric']

df['Bedroom_Num'] = pd.to_numeric(df['Bedroom_Num'], errors='coerce')
df['Bedroom_Num'] = df['Bedroom_Num'].fillna(df['Bedroom_Num'].median())

df['bathrooms_numeric'] = df['bathrooms_numeric'].fillna(df['bathrooms_numeric'].median())


features = ['Size_sqm',
    'Bedroom_Num',
    'bathrooms_numeric',
    'type',
    'location','Price_per_sqft' ]
X = df[features]



df.columns



X = pd.get_dummies(X, columns=['type', 'location'], drop_first=True)

print(f"Final features shape: {X.shape}")
print(f"Target shape: {t.shape}")

X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

model = LinearRegression()
model.fit(X_train, t_train)


y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)


r2_train = r2_score(t_train, y_pred_train)
r2_val = r2_score(t_val, y_pred_val)

print(f"Training R²: {r2_train:.4f}")
print(f"Validation R²: {r2_val:.4f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(t_val, y_pred_val)
rmse = np.sqrt(mean_squared_error(t_val, y_pred_val))
print(f"MAE: {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")

from sklearn.linear_model import Ridge

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, t_train)


y_ridge_pred = ridge_model.predict(X_val)
r2_ridge = r2_score(t_val, y_ridge_pred)
print(f"Ridge Regression R²: {r2_ridge:.4f}")


alphas = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, t_train)
    score = ridge.score(X_val, t_val)
    print(f"Alpha {alpha}: R² = {score:.4f}")

from sklearn.linear_model import Lasso

# Lasso Regression
lasso_model = Lasso(alpha=0.1, max_iter=10000)
lasso_model.fit(X_train, t_train)

y_lasso_pred = lasso_model.predict(X_val)
r2_lasso = r2_score(t_val, y_lasso_pred)
print(f"Lasso Regression R²: {r2_lasso:.4f}")


feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lasso_model.coef_
})
selected_features = feature_importance[feature_importance['coefficient'] != 0]
print(f"Lasso selected {len(selected_features)} features out of {len(X.columns)}")

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Neural Network with Backpropagation
nn_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=1000,
    random_state=42
)

# Train the neural network
nn_model.fit(X_train_scaled, t_train)


y_nn_pred = nn_model.predict(X_val_scaled)
r2_nn = r2_score(t_val, y_nn_pred)
print(f"Neural Network R²: {r2_nn:.4f}")

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, t_train)
y_pred = model.predict(X_val)
r2 = r2_score(t_val, y_pred)
print("R-squared:", r2)

import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1, max_iter=10000),
    'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'XGBoost': xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=4400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

results = {}
for name, model in models.items():
    if name == 'Neural Network':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        model.fit(X_train_scaled, t_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, t_train)
        y_pred = model.predict(X_val)

    r2 = r2_score(t_val, y_pred)
    mae = mean_absolute_error(t_val, y_pred)
    rmse = np.sqrt(mean_squared_error(t_val, y_pred)) # Calculate RMSE for all models
    results[name] = {'R²': r2, 'MAE': mae, 'RMSE': rmse}


results_df = pd.DataFrame(results).T
print("MODEL COMPARISON:")
print(results_df.round(4))


import joblib

# After your model comparison section, add this:

# Find the best model
best_model_name = results_df['R²'].idxmax()
print(f"\n✓ Best Model: {best_model_name} (R² = {results_df.loc[best_model_name, 'R²']:.4f})")

# Get the best model (XGBoost in your case)
best_model = models['XGBoost']

print("\n" + "="*60)
print("SAVING MODEL FILES")
print("="*60)

# 1. Save the trained model
joblib.dump(best_model, 'xgboost_model.pkl')
print("✓ Saved: xgboost_model.pkl")

# 2. Save feature columns (for prediction later)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')
print("✓ Saved: feature_columns.pkl")

# 3. Save model statistics
stats = {
    'bedroom_median': df['Bedroom_Num'].median(),
    'bathroom_median': df['bathrooms_numeric'].median(),
    'r2_score': results['XGBoost']['R²'],
    'mae': results['XGBoost']['MAE'],
    'rmse': results['XGBoost']['RMSE']
}
joblib.dump(stats, 'model_stats.pkl')
print("✓ Saved: model_stats.pkl")

print("\n" + "="*60)
print("MODEL DEPLOYMENT READY!")
print("="*60)
print("Files created:")
print("  - xgboost_model.pkl (trained model)")
print("  - feature_columns.pkl (feature names)")
print("  - model_stats.pkl (model performance)")