# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  # For hypothesis testing
from scipy.optimize import minimize  # For portfolio optimization

# Step 1: File Handling - Reading the CSV file
# Replace 'path/to/your/stock_data.csv' with the actual path to your CSV file
file_path = '.vscode/Date,Symbol,Company,Sector,Price,Vo.csv'
df = pd.read_csv(file_path, parse_dates=['Date'])

# Step 2: Pandas Data Handling
# Set Date as index
df.set_index('Date', inplace=True)

# Handle missing values (if any) - Fill with forward fill for prices, or drop rows
df.fillna(method='ffill', inplace=True)  # Forward fill for continuity in time series
df.dropna(inplace=True)  # Drop any remaining NaNs

# Indexing and Filtering example
# Filter data for Technology sector
tech_df = df[df['Sector'] == 'Technology']

# GroupBy and Aggregation example
# Group by Symbol and calculate average Price and Volume
grouped = df.groupby('Symbol').agg({'Price': 'mean', 'Volume': 'sum'})
print("Grouped Aggregations:\n", grouped.head())

# Merge/Join example (simulating merging with another DataFrame)
# Create a small DataFrame for demonstration
additional_data = pd.DataFrame({
    'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
    'CEO': ['Tim Cook', 'Satya Nadella', 'Sundar Pichai']
})
merged_df = pd.merge(df.reset_index(), additional_data, on='Symbol', how='left')
print("Merged DataFrame sample:\n", merged_df.head())

# Pivot the DataFrame to have Symbols as columns and Prices as values for easier return calculations
price_df = df.pivot_table(values='Price', index=df.index, columns='Symbol')
price_df.fillna(method='ffill', inplace=True)  # Handle any gaps

# Step 3: Calculate Daily Returns using NumPy and Pandas
# Daily returns: (current_price - previous_price) / previous_price
returns = price_df.pct_change().dropna()  # Pandas pct_change for returns

# NumPy operations on returns
returns_np = returns.to_numpy()  # Convert to NumPy array
mean_returns_np = np.mean(returns_np, axis=0)  # Mean returns per asset
std_returns_np = np.std(returns_np, axis=0)  # Standard deviation per asset
cov_matrix_np = np.cov(returns_np.T)  # Covariance matrix
corr_matrix_np = np.corrcoef(returns_np.T)  # Correlation matrix

# Step 4: Statistics - Descriptive Statistics
# Using Pandas describe for mean, std, etc.
desc_stats = returns.describe().T  # Transpose for better view
desc_stats['median'] = returns.median()
desc_stats['mode'] = returns.mode().iloc[0]  # Mode (first mode if multiple)
desc_stats['variance'] = returns.var()
desc_stats['skewness'] = returns.skew()  # Optional
desc_stats['kurtosis'] = returns.kurtosis()  # Optional
print("Descriptive Statistics:\n", desc_stats)

# Correlation and Covariance
cov_matrix = returns.cov()  # Pandas covariance
corr_matrix = returns.corr()  # Pandas correlation
print("Covariance Matrix sample:\n", cov_matrix.iloc[:5, :5])
print("Correlation Matrix sample:\n", corr_matrix.iloc[:5, :5])

# Step 5: Risk & Return Calculations
# Assume equal weights for a sample portfolio (select first 5 symbols for demo)
symbols = returns.columns[:5]  # e.g., ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
num_assets = len(symbols)
weights = np.array([1.0 / num_assets] * num_assets)  # Equal weights

# Expected portfolio return (weighted average of mean returns)
expected_returns = returns[symbols].mean()
portfolio_return = np.dot(weights, expected_returns)

# Portfolio variance (using matrix operations)
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix_np[:num_assets, :num_assets], weights))
portfolio_std = np.sqrt(portfolio_variance)  # Risk (standard deviation)

print(f"Portfolio Expected Return: {portfolio_return:.4f}")
print(f"Portfolio Variance: {portfolio_variance:.4f}")
print(f"Portfolio Std (Risk): {portfolio_std:.4f}")

# Step 6: Portfolio Optimization (Minimize Variance using SciPy)
# Objective function: Portfolio variance
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

# Constraints: Weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Bounds: Weights between 0 and 1 (no short selling)
bounds = tuple((0, 1) for _ in range(num_assets))

# Initial guess: Equal weights
init_weights = [1. / num_assets] * num_assets

# Optimize
opt_result = minimize(portfolio_variance, init_weights,
                      args=(cov_matrix_np[:num_assets, :num_assets],),
                      method='SLSQP', bounds=bounds, constraints=constraints)

optimal_weights = opt_result.x
optimal_return = np.dot(optimal_weights, expected_returns)
optimal_variance = portfolio_variance(optimal_weights, cov_matrix_np[:num_assets, :num_assets])
optimal_std = np.sqrt(optimal_variance)

print("Optimal Weights:", dict(zip(symbols, optimal_weights)))
print(f"Optimal Portfolio Return: {optimal_return:.4f}")
print(f"Optimal Portfolio Std (Risk): {optimal_std:.4f}")

# Step 7: Hypothesis Testing (Optional)
# Example: t-test to compare mean returns of two assets (AAPL vs MSFT)
aapl_returns = returns['AAPL']
msft_returns = returns['MSFT']
t_stat, p_value = stats.ttest_ind(aapl_returns, msft_returns)
print(f"t-test (AAPL vs MSFT returns): t-stat={t_stat:.4f}, p-value={p_value:.4f}")

# Chi-Square test (example: test independence of sectors and high volatility, binned)
# Bin volatility into categories
df['Volatility_Category'] = pd.cut(df['Volatility'], bins=3, labels=['Low', 'Medium', 'High'])
contingency_table = pd.crosstab(df['Sector'], df['Volatility_Category'])
chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi-Square Test (Sector vs Volatility): chi2={chi2:.4f}, p-value={chi_p:.4f}")

# Step 8: Visualizations with Matplotlib and Seaborn
# Heatmap: Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap='coolwarm')  # Sample 10x10
plt.title('Asset Correlation Heatmap')
plt.show()

# Line Plot: Portfolio Growth (cumulative returns for sample portfolio)
cum_returns = (1 + returns[symbols]).cumprod() - 1
portfolio_cum = (cum_returns * weights).sum(axis=1)
plt.figure(figsize=(12, 6))
plt.plot(portfolio_cum, label='Portfolio Cumulative Returns')
plt.title('Portfolio Growth Over Time')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Bar Chart: Asset Allocation (Optimal Weights)
plt.figure(figsize=(8, 5))
plt.bar(symbols, optimal_weights)
plt.title('Optimal Portfolio Asset Allocation')
plt.xlabel('Assets')
plt.ylabel('Weights')
plt.show()

# Pie Chart: Portfolio Composition
plt.figure(figsize=(8, 8))
plt.pie(optimal_weights, labels=symbols, autopct='%1.1f%%')
plt.title('Optimal Portfolio Composition')
plt.show()

# Step 9: Basic File Handling - Writing Results to CSV
# Save descriptive stats to CSV
desc_stats.to_csv('descriptive_stats.csv')
print("Descriptive stats saved to 'descriptive_stats.csv'")