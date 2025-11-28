import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def get_books_from_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    books = []
    book_container = soup.find_all('article', class_='product_pod')
    for book in book_container:
        title = book.find('h3').find('a')['title']
        price = book.find('p', class_='price_color').text
        availability = book.find('p', class_='instock availability').text.strip()
        star_rating = book.find('p', class_='star-rating')['class'][1]     
        book_info = {
            'Title': title,
            'Price': price,
            'Availability': availability,
            'Star Rating': star_rating
        }
        books.append(book_info)
    return books

def scrape_all_books(base_url):
    all_books = []
    page_number = 1
    while True:
        print(f"Scraping page {page_number}...")
        page_url = f"{base_url}/catalogue/page-{page_number}.html"
        books = get_books_from_page(page_url)
        if not books:
            break
        all_books.extend(books)
        page_number += 1
    return all_books

base_url = "http://books.toscrape.com"
all_books = scrape_all_books(base_url)
df = pd.DataFrame(all_books)
folder_path = "Assignment - 4"
os.makedirs(folder_path, exist_ok=True)
csv_file_path = os.path.join(folder_path, 'books.csv')
df.to_csv(csv_file_path, index=False)
print(f"Scrapping completed. Data saved to books.csv")
