#!/usr/bin/env python
# coding: utf-8

# In[57]:


import re
import time
import sqlite3
from selenium import webdriver
from selenium.webdriver.chrome.service import Service

#Step 1: Get the webpage (using webdrive)
s=Service("C:\\Users\\phamd\\Business Programming\\chromedriver.exe")
driver = webdriver.Chrome(service=s)
URL_pattern_str= "https://www.bestbuy.com/site/reviews/nintendo-switch--oled-model-w-neon-red-neon-blue-joy-con-neon-red-neon-blue/6470924?variant=A&page=$NUM$"
time.sleep(5) #Let it become more humanlike
page_content = driver.page_source

page_URL=URL_pattern_str.replace("$NUM$",str(1))
driver.get(page_URL)
page_content = driver.page_source
print("Collecting reviews from: "+ page_URL)

#Parsing the first page to calculate the maximal number of review pages
num_of_pages=0
g=re.compile(r"Rating 4.9 out of 5 stars with (\d*?) reviews",re.S|re.I).findall(page_content)  
if len(g)>0:
    total_num_reviews=g[0].strip().replace(",","")
    print("Total Number of Reviews: " + str(total_num_reviews))
    num_of_pages=int(total_num_reviews)//20+1

#prepare the database for stroing results
conn = sqlite3.connect('bbb.db')
c = conn.cursor()
c.execute("CREATE TABLE Reviews1(              userName varchar(100),               numStar varchar(10),               title varchar(50),               reviewDate varchar(50),               reviewContent text,               numHelpful varchar(10),               numUnhelpful varchar(10))")

#Collecting reviews from all pages
for i in range(1, num_of_pages+1):
    all_chunks=re.compile(r"review-item(.*?)comments-actions",re.S|re.I).findall(page_content)  
    if len(all_chunks)>0: 
        for chunk in all_chunks:

            #print(chunk)
            
            #initialization
            username=""
            num_star=""
            title="" 
            review_date=""
            review_content=""
            num_helpful = ""
            num_unhelpful = ""
            
            #parsing username
            matches=re.compile(r"<strong>(.*?)<\/strong>",re.S|re.I).findall(chunk)  
            if(len(matches)>0):
                username=matches[0]
                
            #parsing num_star
            matches=re.compile(r"Rated (\d) out of 5 stars" ,re.S|re.I).findall(chunk) 
            if(len(matches)>0):
                num_star=matches[0].strip()
                
            #parsing title
            matches =re.compile(r'id="review-id.*?">(.*?)<\/h4>', re.S|re.I).findall(chunk)  
            if(len(matches)>0):
                title=matches[0].strip()
                
            #parsing review date 
            matches =re.compile(r'title=".*?">(.*?)<\/time>', re.S|re.I).findall(chunk)  
            if(len(matches) >0):
                review_date = matches[0].strip()
            
            #parsing review_content
            matches =re.compile(r'<p .*?>(.*?)<\/p>', re.S|re.I).findall(chunk)  
            if(len(matches)>0):
                review_content=matches[1].strip()
                
             #parsing num_helpful
            matches =re.compile(r'Helpful \((.*?)\)<\/button>', re.S|re.I).findall(chunk)  
            if(len(matches)>0):
                num_helpful=matches[0].strip()
                
             #parsing num_unhelpful
            matches =re.compile(r'Unhelpful \((.*?)\)<\/button>', re.S|re.I).findall(chunk)  
            if(len(matches)>0):
                num_unhelpful=matches[0].strip()
             
            # printing collected data to screen
            #print(username +":"+ review_text +":"+ num_star +":"+ country +":"+ review_date +":"+ num_helpful) 
            #Save the extracted data into the database
            query = "INSERT INTO Reviews1 VALUES (?, ?, ?, ?, ?, ?, ?)"
            c.execute(query, (username, num_star, title, review_date, review_content, num_helpful, num_unhelpful))
            
    #Identify the last page
    matches = re.compile(r'<a aria-disabled="true" class=" disabled" data-track="Page next" role="button" title="next Page">', re.S|re.I).findall(page_content)
    if len(matches) > 0:
        break;
    else:
        #get next review page
        page_URL=URL_pattern_str.replace("$NUM$",str(i+1))
        print("Collecting reviews from: "+page_URL)
        time.sleep(1) # pause one second between HTML code
        driver.get(page_URL)
        page_content=driver.page_source   # getting HTML source of page i

conn.commit()
conn.close()

driver.close()
print("\n\nCollection Finished!")  


# In[15]:


import pandas as pd
import sqlite3
conn = sqlite3.connect('bbb.db')
c = conn.cursor()
c.execute("SELECT * from [Reviews1]")
rows=c.fetchall()

header=""
for column_info in c.description:
    header+=column_info[0]+","
print(header)

for row in rows:
    print(row)

conn.close()


df = pd.DataFrame(rows, columns = ['userName', 'numStar', 'title','reviewDate','reviewContent','numHelpful','numUnhelpful'])
print(df)


# In[59]:


# Regression analysis is performed using statsmodels.api, with overall review experience rating as
# the dependent variable (outcome variable) and other collected data fields as independent
# variables (input variables). Results are shown in the table 

# Creating csv file for our data
df.to_csv("bestbuy.csv")


# In[2]:


#Reading raw data from csv
data=pd.read_csv('bestbuy.csv',header=0)
data


# In[60]:


dropped = data.dropna()
dropped
# -> There is 1 row with NA 


# In[61]:


# Replace NA with 0:
data.fillna(0, inplace=True)
data


# In[75]:


# Minaal's part - Descriptive analysis

df1 = df.astype({'numHelpful':'float','numUnhelpful':'float'})
#Top 10 Helpul Reviews
df1.nlargest(n=10, columns=['numHelpful']) 
# Most of top Helpful reviews rate the product 5 star


# In[77]:


#Top 10 Unhelpful Review
df1.nlargest(n=10,columns=['numUnhelpful'])
# Most of top Unhelpful reviews rate the product 5 star


# In[76]:


# 5 Star Reviews for a month ago reviews
Fivestar1 = df[(df['reviewDate'] == "1 month ago") & (df['numStar'] == '5')]
Fivestar1
# 266 5 Star Reviews in a month


# In[4]:


# No of 5 Star Reviews for 2 months ago reviews: 186
Fivestar2 = df[(df['reviewDate'] == "2 months ago") & (df['numStar'] == '5')]
Fivestar2.shape[0]
# No of 5 Star Reviews for 3 months ago reviews: 243
Fivestar3 = df[(df['reviewDate'] == "3 months ago") & (df['numStar'] == '5')]
Fivestar3.shape[0]
# No of 5 Star Reviews for 4 months ago reviews: 259
Fivestar4 = df[(df['reviewDate'] == "4 months ago") & (df['numStar'] == '5')]
Fivestar4.shape[0]
# No of 5 Star Reviews for 5 months ago reviews: 216
Fivestar5 = df[(df['reviewDate'] == "5 months ago") & (df['numStar'] == '5')]
Fivestar5.shape[0]
# No of 5 Star Reviews for 6 months ago reviews: 191
Fivestar6 = df[(df['reviewDate'] == "6 months ago") & (df['numStar'] == '5')]
Fivestar6.shape[0]
#No of 5 Star Reviews for 7 months ago reviews: 267
Fivestar7 = df[(df['reviewDate'] == "7 months ago") & (df['numStar'] == '5')]
Fivestar7.shape[0]
# No of 5 Star Reviews for 8 months ago reviews: 366
Fivestar8 = df[(df['reviewDate'] == "8 months ago") & (df['numStar'] == '5')]
Fivestar8.shape[0]
# No of 5 Star Reviews for 9 months ago reviews: 263
Fivestar9 = df[(df['reviewDate'] == "9 months ago") & (df['numStar'] == '5')]
Fivestar9.shape[0]
# No of 5 Star Reviews for 10 months ago reviews: 530
Fivestar10 = df[(df['reviewDate'] == "10 months ago") & (df['numStar'] == '5')]
Fivestar10.shape[0]
# No of 5 Star Reviews for 11 months ago reviews: 848
Fivestar11 = df[(df['reviewDate'] == "11 months ago") & (df['numStar'] == '5')]
Fivestar11.shape[0]
# No of 5 Star Reviews for 12 months ago reviews: 1227
Fivestar12 = df[(df['reviewDate'] == "1 year ago") & (df['numStar'] == '5')]
Fivestar12.shape[0]


# In[5]:


positive1 = df[(df['reviewDate'] == "1 month ago") & (df['numStar'] >= '3')]
positive1.shape[0]
#No of 5 Star Reviews for 2 months ago reviews
positive2 = df[(df['reviewDate'] == "2 months ago") & (df['numStar'] >= '3')]
positive2.shape[0]
#No of 5 Star Reviews for 3 months ago reviews
positive3 = df[(df['reviewDate'] == "3 months ago") & (df['numStar'] >= '3')]
positive3.shape[0]
#No of 5 Star Reviews for 4 months ago reviews
positive4 = df[(df['reviewDate'] == "4 months ago") & (df['numStar'] >= '3')]
positive4.shape[0]
#No of 5 Star Reviews for 5 months ago reviews
positive5 = df[(df['reviewDate'] == "5 months ago") & (df['numStar'] >= '3')]
positive5.shape[0]
#No of 5 Star Reviews for 6 months ago reviews
positive6 = df[(df['reviewDate'] == "6 months ago") & (df['numStar'] >= '3')]
positive6.shape[0]
#No of 5 Star Reviews for 7 months ago reviews
positive7 = df[(df['reviewDate'] == "7 months ago") & (df['numStar'] >= '3')]
positive7.shape[0]
#No of 5 Star Reviews for 8 months ago reviews
positive8 = df[(df['reviewDate'] == "8 months ago") & (df['numStar'] >= '3')]
positive8.shape[0]
#No of 5 Star Reviews for 9 months ago reviews
positive9 = df[(df['reviewDate'] == "9 months ago") & (df['numStar'] >= '3')]
positive9.shape[0]
#No of 5 Star Reviews for 10 months ago reviews
positive10 = df[(df['reviewDate'] == "10 months ago") & (df['numStar'] >= '3')]
positive10.shape[0]
#No of 5 Star Reviews for 11 months ago reviews
positive11 = df[(df['reviewDate'] == "11 months ago") & (df['numStar'] >= '3')]
positive11.shape[0]

negative1 = df[(df['reviewDate'] == "1 month ago") & (df['numStar'] < '3')]
negative1.shape[0]
#No of 5 Star Reviews for 2 months ago reviews
negative2 = df[(df['reviewDate'] == "2 months ago") & (df['numStar'] < '3')]
negative2.shape[0]
#No of 5 Star Reviews for 3 months ago reviews
negative3 = df[(df['reviewDate'] == "3 months ago") & (df['numStar'] < '3')]
negative3.shape[0]
#No of 5 Star Reviews for 4 months ago reviews
negative4 = df[(df['reviewDate'] == "4 months ago") & (df['numStar'] < '3')]
negative4.shape[0]
#No of 5 Star Reviews for 5 months ago reviews
negative5 = df[(df['reviewDate'] == "5 months ago") & (df['numStar'] < '3')]
negative5.shape[0]
#No of 5 Star Reviews for 6 months ago reviews
negative6 = df[(df['reviewDate'] == "6 months ago") & (df['numStar'] < '3')]
negative6.shape[0]
#No of 5 Star Reviews for 7 months ago reviews
negative7 = df[(df['reviewDate'] == "7 months ago") & (df['numStar'] < '3')]
negative7.shape[0]
#No of 5 Star Reviews for 8 months ago reviews
negative8 = df[(df['reviewDate'] == "8 months ago") & (df['numStar'] < '3')]
negative8.shape[0]
#No of 5 Star Reviews for 9 months ago reviews
negative9 = df[(df['reviewDate'] == "9 months ago") & (df['numStar'] < '3')]
negative9.shape[0]
#No of 5 Star Reviews for 10 months ago reviews
negative10 = df[(df['reviewDate'] == "10 months ago") & (df['numStar'] < '3')]
negative10.shape[0]
#No of 5 Star Reviews for 11 months ago reviews
negative11 = df[(df['reviewDate'] == "11 months ago") & (df['numStar'] < '3')]
negative11.shape[0]


# In[8]:


# Info method to get detail of the data set
data.info()


# In[102]:


# Descriptive analysis:
# As for the following table and all other code:

data[data['numStar']==5] # 5143 
data[data['numStar']==1] # 30
data.describe()

# We can say that there are 5143 customers who vote 5 star account for 90,5% and there are only 30 customers who vote 1 star, 
# which account for 0,5%. The most helpful reviews get 73 reviews and the most unhelpful reviews get 195 unhelpful reviews.
# The average star that Nintendo switch red and blue joycon get is 4.87 star which is really great. Standard deviation of 
# numHelpful is 1.36 and for numUnhelpful is 3.43


# In[22]:


##Yi Yung Chen's part:
# Visualization
import seaborn as sns
df = pd.read_csv("bestbuy.csv")
x = df.index
y = df['numUnhelpful']
plt.figure(figsize = (12,6))
sns.scatterplot(x,y, color='red')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Unhelpfuls')


# In[23]:


import seaborn as sns
df = pd.read_csv("bestbuy.csv")
x = df.index
y = df['numStar']
plt.figure(figsize = (12,6))
sns.scatterplot(x,y, color='red')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Stars')


# In[24]:


##Yi Yung Chen's part
df = pd.read_csv("bestbuy.csv")
x = df.index
y = df['numHelpful']
plt.figure(figsize = (12,6))
sns.scatterplot(x,y, color='red')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Helpfuls')


# In[25]:


##Yi Yung Chen's part
sns.set_theme(style='darkgrid')
plt.figure(figsize = (12,7))

sns.histplot(x=df['numStar'], y=df['numHelpful'],   kde=False, bins=10, color='green')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Helpful')


# In[26]:


##Yi Yung Chen's part
sns.set_theme(style='darkgrid')
plt.figure(figsize = (12,7))

sns.histplot(x=df['numStar'], y=df['numUnhelpful'],   kde=False, bins=10, color='green')
plt.xlabel('Number of Stars')
plt.ylabel('Number of Unhelpful')


# In[27]:


##Yi Yung Chen's part
import matplotlib.pyplot as plt
import numpy as np

y = np.array([positive1.shape[0], positive2.shape[0], positive3.shape[0], positive4.shape[0], positive5.shape[0]
              , positive6.shape[0], positive7.shape[0], positive8.shape[0], positive9.shape[0], positive10.shape[0], positive11.shape[0]])
title = ["last month ago", "2 months ago", "3 months ago", "4 months ago","5 months ago","6 months ago","7 months ago",
            "8 months ago","9 months ago","10 months ago","11 months ago"]
myexplode = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, .2 ]
plt.figure(figsize = (13,13))
plt.title('Positive reviews of each month ago')
plt.pie(y, labels = y, explode = myexplode, shadow = True, autopct='%1.1f%%')
plt.legend(title)
plt.show() 


# In[28]:


import matplotlib.pyplot as plt
import numpy as np

y = np.array([negative1.shape[0], negative2.shape[0], negative3.shape[0], negative4.shape[0], negative5.shape[0]
              , negative6.shape[0], negative7.shape[0], negative8.shape[0], negative9.shape[0], negative10.shape[0], negative11.shape[0]])
title = ["last month ago", "2 months ago", "3 months ago", "4 months ago","5 months ago","6 months ago","7 months ago",
            "8 months ago","9 months ago","10 months ago","11 months ago"]
myexplode = [0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0, 0.2 ]

plt.figure(figsize = (9,9))
plt.title('Negative reviews of each month ago')
plt.pie(y, labels = y, explode = myexplode, shadow = True, autopct='%1.1f%%')
plt.legend(title, loc="upper left")
plt.show() 


# In[34]:


from os import path
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#Put the contents into string
text="".join(row[4] for row in rows)

# Generate a word cloud image
wc = WordCloud(background_color="black")
wc.generate(text)
# Display the generated image using matplotlib
plt.figure(figsize = (12,7))
plt.imshow(wc)
plt.axis("off")
plt.show()


# In[64]:


# Duc's part: Regression Analysis
# Data cleaned:
# Regression model
# Specify independent variable (x= People found helpful) and dependent variables (y= Number of Star) from data
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x=data[['numHelpful']].to_numpy()
y=data[['numStar']].to_numpy()

# plot the raw data x=People found helpful(numHelpful), y= Number of Star(numStar)
plt.figure()
ax = plt.axes()
ax.set_xlabel('People found helpful')
ax.set_ylabel('Number of Star')
ax.scatter(x, y)


# In[65]:


#Create and fit a linear regression model
model = LinearRegression()
model.fit(x, y)


# In[66]:


#check results
model.intercept_
model.coef_


# In[67]:


#check R-squared
model.score(x, y)


# In[68]:


#Use the fitted model to predict estimated crime rate on original poverty (x)
y_est = model.predict(x)

#Plot the estimated line along with scattered raw data on figure
ax=plt.axes()
ax.set_xlabel('numHelpful')
ax.set_ylabel('numStar')
ax.scatter(x, y)
ax.plot(x,y_est)


# In[69]:


#Specify independent variables (x) and dependent variables (y) from data
x=data[['numHelpful','numUnhelpful']].to_numpy()
y=data[['numStar']].to_numpy()


# In[70]:


#Create and fit a linear regression model
model = LinearRegression()
model.fit(x, y)
print("Coefficients:", model.coef_)
print("R-square:", model.score(x, y))


# In[49]:


import statsmodels.api as sm

# Using statsmodels.api to see regression model and make analysis
X= data[['numHelpful','numUnhelpful']].to_numpy()
Y= data[['numStar']].to_numpy()
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.summary())


# In[71]:


# For the OLS Regression Results, we can say that the R-squared is very low, showing that the data we put in does not fit 
# the regression model much (only 3%).
# We have that: numStar predict = 4.87 + (-0.058)*x1 or 4.87 - 0.0015*x2
# While holding numUnhelpful constant, for each time number of people found a comment helpful increase, on average the number of Star will decrease
# While holding numHelpful constant, for each time number of people found a comment unhelpful increase, on average the number of Star will decrease
# In other word, the results show that number of people think a comment is helpful or unhelpful (numHelpful, numUnhelpful)
# negatively affect the number of star.
# p-value is <0,05 so we can say that there is a statistically significant correlation between the independent variables and the
# dependent variable.


# In[6]:


# Denis's part
# Sentinent analysis:
# Read dataframe:
data=pd.read_csv('bestbuy.csv',header=0)
data


# In[7]:


import pandas as pd
import numpy as np
import sqlite3
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
import seaborn as sns
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
stop = stopwords.words('english')
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad
import warnings
warnings.filterwarnings("ignore")


# In[10]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer as vad
# Copying the data into a new dataframe called vader
vader=data.copy()

sentiment=vad()
# Making additional columns for sentiment score in the vader dataframe
sen=['Positive','Negative','Neutral']
sentiments=[sentiment.polarity_scores(i) for i in vader['reviewContent'].values]
vader['Negative Score']=[i['neg'] for i in sentiments]
vader['Positive Score']=[i['pos'] for i in sentiments]
vader['Neutral Score']=[i['neu'] for i in sentiments]
vader['Compound Score']=[i['compound'] for i in sentiments]
score=vader['Compound Score'].values
t=[]
for i in score:
    if i >=0.05 :
        t.append('Positive')
    elif i<=-0.05 :
        t.append('Negative')
    else:
        t.append('Neutral')
vader['Overall Sentiment']=t


# In[11]:


# Having a look at the vader datafram
vader.head()


# In[12]:


import seaborn as sns
sns.countplot(vader['Overall Sentiment'])


# In[13]:


explode = [0, 0.1, 0.1]
vader["Overall Sentiment"].value_counts().plot.pie(title="over all sentiment",autopct='%1.1f%%', 
                        explode = explode
                                 )


# In[18]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
get_ipython().run_line_magic('matplotlib', 'inline')
def wordcloud_draw(data, color = 'black'):
    
    words = ' '.join(data)
    
    cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and not word.startswith('#')
                                and word != 'RT'
                            ])
    
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      max_words=50,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    
    plt.figure(1,figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[19]:


# Wordcloud
# On positive Reviews

df_possitive=vader.loc[(vader['Overall Sentiment'] == "Positive")]
wordcloud_draw( df_possitive['reviewContent'].astype(str),'white')


# In[20]:


# Word Cloud on Negative Review
df_possitive=vader.loc[(vader['Overall Sentiment'] == "Negative")]
wordcloud_draw( df_possitive['reviewContent'].astype(str),'white')


# In[21]:


# Word Cloud on Neutral Reviews
df_possitive=vader.loc[(vader['Overall Sentiment'] == "Neutral")]
wordcloud_draw( df_possitive['reviewContent'].astype(str),'white')


# In[ ]:




