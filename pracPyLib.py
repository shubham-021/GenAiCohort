import requests
from bs4 import BeautifulSoup

BASE_URL = "https://docs.chaicode.com/"
r = requests.get('https://docs.chaicode.com/youtube/getting-started/')

soup = BeautifulSoup(r.text, 'html.parser')

sidebar = soup.find('nav' , {'aria-label' : 'Main'})
top_level_items = sidebar.select('.top-level > li')

course_structure = {}
for item in top_level_items:
    details = item.find('details')
    if details:
        course_name = details.find('span' , class_='large').get_text(strip=True)

        course_links = {}
        for link in details.select('ul a[href]'):
            page_name = link.get_text(strip=True)
            page_url = f"{BASE_URL}{link['href']}"
            course_links[page_name] = page_url

        course_structure[course_name] = course_links

print(course_structure.items())