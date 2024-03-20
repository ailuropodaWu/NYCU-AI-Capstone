# Create Dateset
## About dataset
This is a dataset on the men’s 100 meters events in the Olympics from 1948 to 2020 (without 1952 since there is no wind information from that year).
### Attributes
The attributes in this dataset correspond to the information about athletes and other relevant details about the events. The label of the dataset represents the performance of each athlete in a competition.
Each row represents a record with following attributes about a competitor in a single game. And here is the list of attributes:
- ***Name***:           string, name of the competitor, unuseful feature
- ***Nation***:         int, code of the nation where the competitor from according to the dictionary you can see at [generate_train_data.ipynb](https://github.com/ailuropodaWu/NYCU-AI-Capstone/blob/main/Project1/generate_train_data.ipynb)
- ***Weight*** *(kg)*:  float, weight of the competitor
- ***Height*** *(m)*:   float, height of the competitor
- ***BMI***:            float, calculated by the formula 
- ***Age***:            float, age of the competitor, calculated from the birthday to the first day the event start
- ***Year***:           int, the year of the event
- ***Round***:          int, round in the competition, range from 1-4(4 means final)
- ***Wind*** *(m/s)*:   float, wind information of the game
- ***isHometown***:     bool, whether the venue of the event is the hometown of the competitor or not
- ***Label*** *(s)*:    float, performance of the competitor in the game
### Data scraping

### Data processing
### Referrence
## Algoritms
### Supervised learning
### Unsupervised learning