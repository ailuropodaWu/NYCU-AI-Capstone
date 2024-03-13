import requests as rq
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from tqdm import tqdm
data = []
scrape = True

if not scrape:
    df = pd.read_csv('./dataset/raw_train_data.csv')
else:
    for page in tqdm(range(1,181)):
        url = f'https://worldathletics.org/records/all-time-toplists/sprints/100-metres/all/men/senior?regionType=world&timing=electronic&windReading=regular&page={page}&bestResultsOnly=false&firstDay=1900-01-01&lastDay=2016-12-31&maxResultsByCountry=all&eventId=10229630&ageCategory=senior'

        html = rq.get(url).text
        soup = BeautifulSoup(html, 'html.parser')

        table = soup.find('table')

        for row in table.find_all('tr'):
            cols = row.find_all('td')
            # Extracting the table headers
            if len(cols) == 0:
                if page == 1:
                    cols = row.find_all('th')
                else:
                    continue
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])

    headers = data.pop(0)
    df = pd.DataFrame(data, columns=headers)
    df.to_csv('./dataset/raw_train_data.csv')
print(df)
