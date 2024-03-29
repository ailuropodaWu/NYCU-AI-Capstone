import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

# scrape the url of the men's 100m of 1948 - 2020
url_root = "https://www.olympedia.org/"
data = []
wind_list = []
header = []

url_men100 = url_root + "/event_names/40"
url_list_men100 = []
html = rq.get(url_men100).text
soup = BeautifulSoup(html, 'html.parser')
table = soup.find('table', {"class": "table table-striped"})
for row in table.find_all('tr'):
    cols = row.find_all('td')
    if len(cols) == 0 or int(cols[0].text) < 1948:
        continue
    url_list_men100.append(cols[1].a.get('href'))

# scrape the data from the url of the men's 100m on each years
for url_idx in tqdm(range(len(url_list_men100))):
    url = url_list_men100[url_idx]
    html = rq.get(url_root + url).text
    soup = BeautifulSoup(html, 'html.parser')
    wind_info = []

    while True: # Get the information of wind
        t_wind = soup.find('table', {'class': 'biodata'})
        if t_wind is None:
            break
        for row in t_wind.find_all('tr'):
            head = row.find('th').text
            if head != "Wind":
                continue
            wind = row.find('td').text.replace('m/s', '').replace('<', '')
            wind = float(wind) 
            wind_info.append(wind)
            break
        t_wind.decompose()
    
# get the record of all athletes
    table = soup.find('table', {"class": "table table-striped"})
    for row in table.find_all('tr'):
        cols = row.find_all('td')
        if len(cols) == 0: 
            continue
        if url_idx < 16 and cols[4].text.strip()[0] ==  '–':
            # since 2012 the rule of Preliminary Round is changed
            break
        cols = cols[2:] # since the first 2 columns are Pos & Nr

# scrape data of athlete
        url_athlete = cols[0].a.get('href')
        html_athlete = rq.get(url_root + url_athlete).text
        soup_athlete = BeautifulSoup(html_athlete, 'html.parser')
        t_athlete = soup_athlete.find('table', {'class': 'biodata'})
        athlete_info = [0] * 2 # to record birthday and weight&height
        for r in t_athlete.find_all('tr'):
            if r.find('th').text.strip() == 'Born':
                athlete_info[0] = r.find('td')
            if r.find('th').text.strip() == 'Measurements':
                athlete_info[1] = r.find('td')

        cols += athlete_info
        for i in range(len(cols)):
            try:
                cols[i] = cols[i].text.strip()
            except:
                cols[i] = None
        cols.append(url_idx * 4 + 1948)
        data.append(cols)
    wind_list.append(wind_info)

# manually add header and save the data as .csv
header = ['Name', 'Nation', 'R1', 'R2', 'R3', 'R4', 'Gold', 'Silver', 'Bronze', 'Birth', 'Body', 'Year']
df = pd.DataFrame(data, columns=header)
pd.to_pickle(wind_list, './dataset/wind_list.pkl')
df.to_csv('./dataset/raw_data.csv', index=False)

