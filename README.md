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

After concating the data that was collected I stored the data into MySQL so I could reference on the backend.

``` python
hitting_log.to_sql( "snhu_hitting" , con = engine, if_exists = 'append', chunksize = 3321)
pitching_log.to_sql( "snhu_pitching" , con = engine, if_exists = 'append', chunksize = 3321)
```

After the data was stored I had to explore to see what data I had to clean. There were several places to clean including:
   * Wrong data types - Date 
   * NA values for neutral sites
   * Target values split
 
 So i created a small function for it be cleansed using pandas libraries 
  
 ``` python 
 def clean(df):    
    df = df[df['Date'] != 'Total']
    df.fillna(value="neutral", inplace=True)
    df['snhu_result'] = df['W/L'].str.split('-', expand = True)
    df['Date'] = df.Date.str.replace('/','-')
    df['Date'] =pd.to_datetime(ddd['Date'])

    return df
```
### Feature Engineering 

Next I wanted to create some aggregated feautures that would sum up the data:
 * 10 game avg 
 * Batting Avg (By team)
 * Avg opp error
 * Avg era by team 
 * Location
 * AVG HR

Below I had to do a join to get each unique element to match up properly. I was able to do this by joining (inner join) on date,score, and opponent. This allowed me to manipulate the data I wanted to in pandas by added:

* 10 Game BA
* Rolling BA
* Running opp error 
* Avg era 
* Location 
* Avg Rolling HR

These would be then used as features into my XGBoost Model 

``` python
data_hitting = """SELECT DISTINCT
    hitting_snhu.Opponent,
    hitting_snhu.level_0 as `index`, 
    hitting_snhu.`Date`, 
    hitting_snhu.Loc, 
    hitting_snhu.AB, 
    hitting_snhu.R, 
    hitting_snhu.H, 
    hitting_snhu.HR, 
    hitting_snhu.E as Opp_E,  
    hitting_snhu.snhu_result,  
    pitching_snhu.R as SNHU_R,
    pitching_snhu.IP
FROM college_stats.hitting_snhu 
INNER JOIN college_stats.pitching_snhu 
    ON hitting_snhu.`Date` = pitching_snhu.`Date` 
    AND hitting_snhu.score = pitching_snhu.score 
    AND hitting_snhu.Opponent = pitching_snhu.Opponent 
LIMIT 0, 50000;"""

raw_features = pd.read_sql(data_hitting , con = engine)

# Convert object columns to float
raw_features['AB'] = raw_features['AB'].astype(float)
raw_features['H'] = raw_features['H'].astype(float)
raw_features['IP'] = raw_features['IP'].astype(float)
raw_features['R'] = raw_features['R'].astype(float)
# Calculate 10 game averages
raw_features['10 Game Avg'] = raw_features['H'].rolling(10, min_periods=1).mean().round(3)

# Calculate running avg opp error
raw_features['Running Opp Error'] = raw_features['Opp_E'].expanding().mean().round(3)

# Calculate batting average (H/AB)
raw_features['Batting Avg'] = (raw_features['H'] / raw_features['AB']).round(3)

```
## Modeling 

After storing the manipulated data into MySQL I was able to test my XGBoost Model (returned best scores). After Preproccesing my data I Modelled and got these results:

* XGBoost returned a 97.82% accuracy rate with 2.18% error rate with no optimizations built into it and 99 % AUC 
* Our cross validation score (5 fold) gave us 0.97680764 0.98499318 0.97680764 0.96998636 0.97680764 for the results showing that the model indicates low variance 
    

``` python
from sklearn.model_selection import cross_val_score

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define XGBoost model
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, n_estimators = 100)

# Perform 5-fold cross-validation on training set
scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print("Cross-validation scores:", scores)

# Train XGBoost model on full training set
xgb_model.fit(X_train, y_train)

# Evaluate XGBoost model on testing set
y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

```
We can notice at around tree 300 a decrease in variance within the model. But the model does have a large seperation between the two points indicating some bias in the XGboost model.  

![Image 4-23-23 at 7 15 PM](https://user-images.githubusercontent.com/94020684/233871337-923309c7-6a8e-42c9-b5bf-f1b876f1c47c.jpg)

### ROC Curve / AUC score 
- ROC is used for binary classification algos that takes in the specificity and sensitive/recall and plots them agianst each other showing the optimal threshold for classification 
- AUC is the measure of area under the curve based off the ROC 

![download](https://user-images.githubusercontent.com/94020684/233871367-baf4759b-6705-40f2-8090-7c2810c8c5ae.png)

Confusion Matrix:
 [[299   7]
 [ 12 599]]
 
Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.98      0.97       306
           1       0.99      0.98      0.98       611

    accuracy                           0.98       917
   macro avg       0.97      0.98      0.98       917
weighted avg       0.98      0.98      0.98       917


![image](https://user-images.githubusercontent.com/94020684/233863619-f715f829-cd24-4546-8ac7-f9049408a247.png)


![image](https://user-images.githubusercontent.com/94020684/233863625-f45a5948-b787-44f4-aeda-08f1b373a62a.png)








