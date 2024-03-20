# Create my own dataset
## About dataset
This is a dataset on the men’s 100 meters events in the Olympics from 1948 to 2020 (without 1952 since there is no wind information from that year).

### Attributes
The attributes in this dataset correspond to the information about athletes and other relevant details about the events. The label of the dataset represents the performance of each athlete in a competition.
Each row represents a record with following attributes about a competitor in a single game. And here is the list of attributes:
- ***Name***: <u>string</u>, name of the competitor, unuseful feature
- ***Nation***: <u>int</u>, code of the nation where the competitor from according to the dictionary you can see at [generate_train_data.ipynb](https://github.com/ailuropodaWu/NYCU-AI-Capstone/blob/main/Project1/generate_train_data.ipynb)
- ***Weight*** *(kg)*: <u>float</u>, weight of the competitor
- ***Height*** *(m)*: <u>float</u>, height of the competitor
- ***BMI***: <u>float</u>, calculated by the formula $ \dfrac{Height(n)^2}{Weight(kg)} $
- ***Age***: <u>float</u>, age of the competitor, calculated from the birthday to the first day the event start
- ***Year***: <u>int</u>, the year of the event
- ***Round***: <u>int</u>, round in the competition, range from 1-4(4 means final)
- ***Wind*** *(m/s)*: <u>float</u>, wind information of the game
- ***isHometown***: <u>bool</u>, whether the venue of the event is the hometown of the competitor or not
- ***Label*** *(s)*: <u>float</u>, performance of the competitor in the game


### Data scraping

### Data processing
### Referrence
## Algoritms
### Supervised learning
### Unsupervised learning