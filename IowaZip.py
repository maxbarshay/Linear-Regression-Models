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
# print(res)

for i in range(0, len(res), 3):

    val = res[i][-3]

    #print(val)
    if val == "TIE":
        #print("do stuff")

        and_count = 0
        rel = res[i+1]

        to_add = []

        for j in range(len(rel)):


            if re.search(r'f=\"(\d+)', rel[j]) is not None:

                to_add.append(re.search(r'f=\"(\d+)', rel[j]).group(1))

        popspot = res[i+2][1]
        pop = re.search(r"\d+,?\d+", popspot).group()

        for zipcode in to_add:

            zipPops[int(zipcode)] = pop


    else:
        
    # if i % 3 == 2:
        codespot = res[i+1][2]
        code = re.search(r"\d+", codespot).group()
        popspot = res[i+2][1]
        pop = re.search(r"\d+,?\d+", popspot).group()
        zipPops[int(code)] = pop



#
print(zipPops)
print(len(zipPops))
# print(soup)


