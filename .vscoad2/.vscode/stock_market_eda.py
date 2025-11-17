import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
df = pd.read_csv('.vscode/.vscoad2/.vscode/Date,Ticker,Open,High,Low,Close,Vol.csv')

print("="*80)
print("STOCK MARKET EXPLORATORY DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. DATA OVERVIEW & QUALITY ASSESSMENT
# ============================================================================
print("\n" + "="*80)
print("1. DATA OVERVIEW & QUALITY ASSESSMENT")
print("="*80)

print("\nDataset Shape:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nColumn Names and Data Types:")
print(df.dtypes)

print("\nFirst 5 Rows:")
print(df.head())

print("\nLast 5 Rows:")
print(df.tail())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(f"Number of duplicates: {df.duplicated().sum()}")

print("\nBasic Statistical Summary:")
print(df.describe())

print("\nUnique Values:")
print(f"Unique Tickers: {df['Ticker'].nunique()}")
print(f"Tickers: {df['Ticker'].unique()}")
print(f"Unique Sectors: {df['Sector'].nunique()}")
print(f"Sectors: {df['Sector'].unique()}")
print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================
print("\n" + "="*80)
print("2. DATA PREPARATION")
print("="*80)

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Ticker', 'Date'])

# Calculate Daily Returns
df['Daily_Return'] = df.groupby('Ticker')['Close'].pct_change() * 100

# Calculate Price Range
df['Price_Range'] = df['High'] - df['Low']

# Calculate Price Change
df['Price_Change'] = df['Close'] - df['Open']

print("\nNew Features Created:")
print("- Daily_Return: Percentage change in closing price")
print("- Price_Range: Difference between High and Low")
print("- Price_Change: Difference between Close and Open")

print("\nUpdated Dataset Info:")
print(df.info())

# ============================================================================
# 3. PRICE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. PRICE ANALYSIS")
print("="*80)

# Overall statistics by ticker
price_stats = df.groupby('Ticker').agg({
    'Close': ['mean', 'min', 'max', 'std'],
    'Daily_Return': ['mean', 'std'],
    'Volume': 'mean'
}).round(2)

print("\nPrice Statistics by Ticker:")
print(price_stats)

# Performance from start to end
performance = df.groupby('Ticker').agg({
    'Close': ['first', 'last']
}).round(2)
performance.columns = ['Start_Price', 'End_Price']
performance['Change_$'] = (performance['End_Price'] - performance['Start_Price']).round(2)
performance['Change_%'] = ((performance['End_Price'] - performance['Start_Price']) / performance['Start_Price'] * 100).round(2)
performance = performance.sort_values('Change_%', ascending=False)

print("\nStock Performance (Period Return):")
print(performance)

# ============================================================================
# 4. VOLATILITY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. VOLATILITY ANALYSIS")
print("="*80)

volatility = df.groupby('Ticker')['Daily_Return'].std().sort_values(ascending=False)
print("\nVolatility (Std Dev of Daily Returns):")
print(volatility)

print(f"\nMost Volatile Stock: {volatility.idxmax()} ({volatility.max():.2f}%)")
print(f"Least Volatile Stock: {volatility.idxmin()} ({volatility.min():.2f}%)")

# ============================================================================
# 5. VOLUME ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("5. VOLUME ANALYSIS")
print("="*80)

volume_stats = df.groupby('Ticker')['Volume'].agg(['mean', 'min', 'max']).sort_values('mean', ascending=False)
print("\nTrading Volume Statistics:")
print(volume_stats)

# ============================================================================
# 6. SECTOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("6. SECTOR ANALYSIS")
print("="*80)

sector_performance = df.groupby('Sector').agg({
    'Close': 'mean',
    'Volume': 'mean',
    'Daily_Return': 'mean'
}).sort_values('Daily_Return', ascending=False)

print("\nAverage Performance by Sector:")
print(sector_performance)

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. CORRELATION ANALYSIS")
print("="*80)

# Create pivot table for correlation
price_pivot = df.pivot_table(values='Close', index='Date', columns='Ticker')
correlation_matrix = price_pivot.corr()

print("\nCorrelation Matrix (Stock Prices):")
print(correlation_matrix.round(2))

# ============================================================================
# 8. VISUALIZATIONS (Separate Clear Charts)
# ============================================================================
print("\n" + "="*80)
print("8. GENERATING VISUALIZATIONS")
print("="*80)

# Visualization 1: Price Trends Over Time
print("\nGenerating Chart 1: Price Trends Over Time...")
plt.figure(figsize=(14, 7))
for ticker in df['Ticker'].unique():
    ticker_data = df[df['Ticker'] == ticker]
    plt.plot(ticker_data['Date'], ticker_data['Close'], label=ticker, linewidth=2.5, marker='o', markersize=4)
plt.title('Stock Price Trends Over Time', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.ylabel('Closing Price ($)', fontsize=12, fontweight='bold')
plt.legend(fontsize=10, loc='best', frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('01_price_trends.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '01_price_trends.png'")
plt.close()

# Visualization 2: Trading Volume by Stock
print("Generating Chart 2: Trading Volume by Stock...")
plt.figure(figsize=(12, 7))
avg_volume = df.groupby('Ticker')['Volume'].mean().sort_values(ascending=False)
colors = plt.cm.viridis(np.linspace(0, 1, len(avg_volume)))
bars = plt.bar(avg_volume.index, avg_volume.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Average Trading Volume by Stock', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Ticker', fontsize=12, fontweight='bold')
plt.ylabel('Average Volume (Millions)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height/1e6)}M', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('02_trading_volume.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '02_trading_volume.png'")
plt.close()

# Visualization 3: Stock Performance (% Change)
print("Generating Chart 3: Stock Performance...")
plt.figure(figsize=(12, 7))
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in performance['Change_%']]
bars = plt.barh(performance.index, performance['Change_%'], color=colors, edgecolor='black', linewidth=1.5)
plt.title('Stock Performance - Percentage Change', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Percentage Change (%)', fontsize=12, fontweight='bold')
plt.ylabel('Ticker', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(axis='x', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f' {width:.2f}%', ha='left' if width > 0 else 'right', 
             va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('03_stock_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '03_stock_performance.png'")
plt.close()

# Visualization 4: Volatility Comparison
print("Generating Chart 4: Volatility Comparison...")
plt.figure(figsize=(12, 7))
volatility_sorted = volatility.sort_values(ascending=True)
colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(volatility_sorted)))
bars = plt.barh(volatility_sorted.index, volatility_sorted.values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Stock Volatility - Standard Deviation of Returns', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Standard Deviation (%)', fontsize=12, fontweight='bold')
plt.ylabel('Ticker', fontsize=12, fontweight='bold')
plt.grid(axis='x', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f' {width:.2f}%', ha='left', va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('04_volatility.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '04_volatility.png'")
plt.close()

# Visualization 5: Correlation Heatmap
print("Generating Chart 5: Correlation Heatmap...")
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=2, cbar_kws={"shrink": 0.8},
            annot_kws={'fontsize': 11, 'fontweight': 'bold'},
            vmin=-1, vmax=1, center=0)
plt.title('Stock Price Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig('05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '05_correlation_heatmap.png'")
plt.close()

# Visualization 6: Distribution of Daily Returns (Box Plot)
print("Generating Chart 6: Distribution of Daily Returns...")
plt.figure(figsize=(14, 7))
df_clean = df.dropna(subset=['Daily_Return'])
box_data = [df_clean[df_clean['Ticker'] == ticker]['Daily_Return'].values 
            for ticker in df['Ticker'].unique()]
bp = plt.boxplot(box_data, labels=df['Ticker'].unique(), patch_artist=True,
                 notch=True, showmeans=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(bp['boxes'])))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_edgecolor('black')
    patch.set_linewidth(1.5)
plt.title('Distribution of Daily Returns by Stock', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Ticker', fontsize=12, fontweight='bold')
plt.ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
plt.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.xticks(rotation=45, fontsize=11)
plt.tight_layout()
plt.savefig('06_returns_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '06_returns_distribution.png'")
plt.close()

# Visualization 7: Sector Performance
print("Generating Chart 7: Sector Performance...")
plt.figure(figsize=(12, 7))
sector_avg = df.groupby('Sector')['Daily_Return'].mean().sort_values()
colors_sector = ['#2ecc71' if x > 0 else '#e74c3c' for x in sector_avg]
bars = plt.barh(sector_avg.index, sector_avg.values, color=colors_sector, 
                edgecolor='black', linewidth=1.5)
plt.title('Average Daily Return by Sector', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Average Daily Return (%)', fontsize=12, fontweight='bold')
plt.ylabel('Sector', fontsize=12, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(axis='x', alpha=0.3, linestyle='--')
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2.,
             f' {width:.3f}%', ha='left' if width > 0 else 'right', 
             va='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('07_sector_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '07_sector_performance.png'")
plt.close()

# Visualization 8: Price Range Analysis
print("Generating Chart 8: Price Range Analysis...")
plt.figure(figsize=(12, 7))
price_range_avg = df.groupby('Ticker')['Price_Range'].mean().sort_values(ascending=False)
colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(price_range_avg)))
bars = plt.bar(price_range_avg.index, price_range_avg.values, color=colors, 
               edgecolor='black', linewidth=1.5)
plt.title('Average Daily Price Range by Stock', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Ticker', fontsize=12, fontweight='bold')
plt.ylabel('Average Price Range ($)', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, fontsize=11)
plt.grid(axis='y', alpha=0.3, linestyle='--')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'${height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('08_price_range.png', dpi=300, bbox_inches='tight')
print("✓ Saved as '08_price_range.png'")
plt.close()

print("\n✓ All 8 visualizations saved as separate PNG files!")
print("  Files: 01_price_trends.png to 08_price_range.png")

# ============================================================================
# 9. KEY INSIGHTS & FINDINGS
# ============================================================================
print("\n" + "="*80)
print("9. KEY INSIGHTS & FINDINGS")
print("="*80)

print("\n📈 BEST PERFORMERS:")
top_performers = performance.nlargest(3, 'Change_%')
for idx, row in top_performers.iterrows():
    print(f"   {idx}: +{row['Change_%']:.2f}% (${row['Start_Price']:.2f} → ${row['End_Price']:.2f})")

print("\n📉 WORST PERFORMERS:")
worst_performers = performance.nsmallest(3, 'Change_%')
for idx, row in worst_performers.iterrows():
    print(f"   {idx}: {row['Change_%']:.2f}% (${row['Start_Price']:.2f} → ${row['End_Price']:.2f})")

print("\n⚡ MOST VOLATILE STOCKS:")
for ticker in volatility.nlargest(3).index:
    print(f"   {ticker}: {volatility[ticker]:.2f}% std dev")

print("\n🔄 HIGHEST TRADING VOLUME:")
for ticker in volume_stats.nlargest(3, 'mean').index:
    print(f"   {ticker}: {volume_stats.loc[ticker, 'mean']:,.0f} shares avg")

print("\n🔗 HIGHLY CORRELATED PAIRS:")
corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_pairs.append({
            'Stock1': correlation_matrix.columns[i],
            'Stock2': correlation_matrix.columns[j],
            'Correlation': correlation_matrix.iloc[i, j]
        })
corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', ascending=False)
print(corr_df.head(3).to_string(index=False))

print("\n" + "="*80)
print("EDA COMPLETE! 🎉")
print("="*80)
print("\nAll visualizations have been generated and saved.")
print("Check 'stock_market_eda.png' for comprehensive visual analysis.")

plt.show()