import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

url = "https://www.timeanddate.com/weather/"
response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
allCities = []

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", class_="zebra")
    rows = table.find_all("tr")
    for row in rows:
        city_tag = row.find("a")
        temp_tag = row.find("td", class_="rbi")
        condition_tag = row.find("td", class_="r").find("img") if row.find("td", class_="r") else None
        if not city_tag:
            continue
        city_name = city_tag.text.strip()
        temperature = temp_tag.text.strip() if temp_tag else "N/A"
        condition = condition_tag.get("title") if condition_tag else "N/A"
        print(f"City: {city_name}\nTemperature: {temperature}\nCondition: {condition}\n")
        allCities.append({
            "City Name": city_name,
            "Temperature": temperature,
            "Weather Condition": condition
        })

df = pd.DataFrame(allCities)
folder_path = "Assignment - 4"
os.makedirs(folder_path, exist_ok=True)
csv_file_path = os.path.join(folder_path, 'weather.csv')
df.to_csv(csv_file_path, index=False)