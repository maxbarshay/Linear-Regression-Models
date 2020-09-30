from bs4 import BeautifulSoup
import requests
import re

url = "https://www.iowa-demographics.com/zip_codes_by_population"
myRequest = requests.get(url)
soup = BeautifulSoup(myRequest.text,"html.parser")

zipPops = {}

# for code in soup.find_all('a'):
#     text = code.text
#     if text[0].isdigit():
#         print(text)

res = []


for code in soup.find_all('tr'):

    inner = code.find_all('td')

    if len(inner) == 0:
        continue
    else:
        for item in inner:
            item = str(item).split()
            res.append(item)
            

            # print(item)
        

res = res[:-1]
print(res)


for i in range(len(res)):


    if i % 3 == 2:

        prev = res[i-1][2]
        code = re.search(r"\d+", prev).group()
        curr = res[i][1]
        pop = re.search(r"\d+,?\d+", curr).group()
        zipPops[int(code)] = pop


print(zipPops)

# print(soup)