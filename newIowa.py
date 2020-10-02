from bs4 import BeautifulSoup
import requests
import re

url = "https://www.iowa-demographics.com/zip_codes_by_population"
myRequest = requests.get(url)
soup = BeautifulSoup(myRequest.text,"html.parser")

zipPops = {}

i = 0

for code in soup.find_all('tr'):

    first = code.find_all('td')


    i += 1

    if i > 10:
        break
    #print(code)