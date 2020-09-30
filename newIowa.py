from bs4 import BeautifulSoup
import requests
import re

url = "https://www.iowa-demographics.com/zip_codes_by_population"
myRequest = requests.get(url)
soup = BeautifulSoup(myRequest.text,"html.parser")

zipPops = {}

for code in soup.find_all('tr'):


    print(code)