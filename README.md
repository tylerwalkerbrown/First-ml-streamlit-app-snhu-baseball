# First-ml-streamlit-app-snhu-baseball

### Project Goal 
For my first application I decided to do it on my old collegiate baseball team. For this project the goal was to come up with an interface backed my machine learning to predict wins and loses for Southern New Hampshire University. 

### Collecting the Essential Data

The first step in this project was to collect the data I needed. To enable this I created a webscraping script and optimized the data search through an iterative process with the code below. This code does:

* Takes in the SNHU link to format the link to be collected later 
* Iterates over years 2000 - 2023 
* Indexes HTML Tables pertaining to:
    * Hitting 
    * Pitching 
* Appends the links list to be used later             


''''
```python
import pandas as pd

links = []
hitting_log = []
pitching_log = []

for year in range(2000, 2023):
    links.append('https://snhupenmen.com/sports/baseball/stats/{}'.format(year))
    for link in links:
        try:
            hitting_log.append(pd.read_html(link, header=0)[6])
            pitching_log.append(pd.read_html(link, header=0)[7])
        except:
            pass
            
```            
# end of Python code



![image](https://user-images.githubusercontent.com/94020684/233863619-f715f829-cd24-4546-8ac7-f9049408a247.png)


![image](https://user-images.githubusercontent.com/94020684/233863625-f45a5948-b787-44f4-aeda-08f1b373a62a.png)








