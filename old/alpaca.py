import requests

url = "https://data.alpaca.markets/v2/stocks/bars?timeframe=1Min&limit=1000&adjustment=raw&feed=iex&sort=desc&symbols=spy"

headers = {"accept": "application/json", "APCA-API-KEY-ID": "PKGOHA7XQLPC5VNTOJRR", "APCA-API-SECRET-KEY": "pLIfvfHGxShxq601QLG4hFyYxu1I4W96IQLdHPze"}

response = requests.get(url, headers=headers)

print(response.text)