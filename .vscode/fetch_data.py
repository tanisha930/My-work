import requests

url = "https://api.twelvedata.com/time_series"

params =  {
    "symbol": "EUR/USD",
    "interval": "1h",
    "start_date": "2025-08-01",
    "end_date": "2026-08-01",
    "apikey": "4ddd7168fd1e4b0d9141e1cf8fd58c16"}
    

try:
    response = requests.get(url, params=params, timeout=20)

    print("Status code:", response.status_code)
    print("URL:", response.url)
    print("Response:")
    print(response.text)

except requests.exceptions.ConnectionError as e:
    print("Connection error:")
    print(e)

except requests.exceptions.Timeout:
    print("Request timed out.")
    response = requests.get(url)
data = response.json()

print(data)
import pandas as pd

values = data["values"]

df = pd.DataFrame(values)

df["datetime"] = pd.to_datetime(df["datetime"])

price_columns = ["open", "high", "low", "close"]
df[price_columns] = df[price_columns].astype(float)

df = df.sort_values("datetime").reset_index(drop=True)

print(df)
print(df.dtypes)

print("Total rows:", len(df))
print("Oldest:", df["datetime"].min())
print("Newest:", df["datetime"].max())

# Save as CSV
df.to_csv("eurusd_1h_raw.csv", index=False)

print("CSV file saved successfully!")