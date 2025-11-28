import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import os

def init_driver():
    options = webdriver.ChromeOptions()
    options.headless = True
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrape_imdb_top250():
    url = "https://www.imdb.com/chart/top/"
    driver = init_driver()
    driver.get(url)
    time.sleep(3)
    movies = []
    movie_rows = driver.find_elements(By.CSS_SELECTOR, '.lister-list tr')
    for row in movie_rows:
        rank = row.find_element(By.CSS_SELECTOR, '.titleColumn').text.split()[0]
        title = row.find_element(By.CSS_SELECTOR, '.titleColumn a').text
        year = row.find_element(By.CSS_SELECTOR, '.titleColumn span').text.strip("()")
        rating = row.find_element(By.CSS_SELECTOR, '.imdbRating strong').text
        movies.append({
            'Rank': rank,
            'Movie Title': title,
            'Year of Release': year,
            'IMDB Rating': rating
        })
    driver.quit()
    return movies

print("Scraping IMDB Top 250 Movies...")
movie_data = scrape_imdb_top250()
df = pd.DataFrame(movie_data)
folder_path = "Assignment - 4"
os.makedirs(folder_path, exist_ok=True)
csv_file_path = os.path.join(folder_path, 'imdb_top250.csv')
df.to_csv(csv_file_path, index=False)
print("Data saved to imdb_top250.csv")
