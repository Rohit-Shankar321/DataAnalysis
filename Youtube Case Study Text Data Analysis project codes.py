#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


# # choosing youtube for data analysis bcoz it's the 2nd most visiting site in the world . It attracts about 44% of all the internet users . 1 billion hours of youtube content is watched per day . 37% of all mobile internet traffic goes to the youtube.

# # In this project I am going to do , 1.Sentiment Analysis.  / 2. Emoji Analysis.  / 3. Dislikes vs Views Analysis.  /4. Trending video Analysis.

# # 1..................................Sentiment Analysis.....................................................

# In[3]:


comments_df = pd.read_csv('D:/Datasets/Data Analysis Projects/UScomments.csv', error_bad_lines = False)
#Here I am specifying the error_bad_lines bcoz csv use comma as delimiter but sometimes in another dataset there will be another kind of delimiter like \t


# In[4]:


print(comments_df.head())


# # While doing the Sentiment Analysis there are two key aspects 1. Polarity and 2. Subjectivity . For eg - I like this video - We have a positive polarity with respect to this sentence ,So our polarity value will be +1 , Polarity ranges between -1 t0 +1.    /   Eg2 - I am going to the market. This sentence conatins no sentiment it conatins subject . Its part of the subjectivity.
#     

# # There are lot of ways in python to examine sentiment analysis . 1. Text Blob , 2. Vader, 3.Spacy . I am using TextBlob - typically NLP library that us built on NLTK  which is all about NAtural Language Processing Toolkit

# # Checking in my Dataset whether there will be any missing value or not.

# In[5]:


comments_df.isnull().sum()        #calling our inbuilt function isnull().sum()


# # We got 25 missing values , Dropping those missing values

# In[6]:


comments_df.dropna(inplace = True)


# In[7]:


print(comments_df.head())


# In[8]:


get_ipython().system('pip install textblob         #textblob successfully installed for sentiment analysis. We can also install it by Anaconda promp using .. conda install textblob')


# In[9]:


from textblob import TextBlob


# In[10]:


TextBlob('trending 😉').sentiment    #lets pass a comment to show how it works 


# In[11]:


TextBlob('trending 😉').sentiment.polarity  #printing the polarity


# # so for the entire comments_df we have to find the polarity for the Sentiment Analysis

# In[12]:


"""
polarity =[]        #created an empty list to store the polarity values

for comment in comments_df['comment_text']: 
    try:                                                    #If any error comeup I am using exception handling to overcome
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)
"""


# # upper data will take a lot of time because we have approx 7 lakh comment . So specifying a limited range of the comments_df

# In[13]:


df = comments_df[0:10000] #taking first 10000 and assigning into a df   , Basically its a slicing inside a new df that I created


# In[14]:


polarity =[]        #created an empty list to store the polarity values

for comment in df['comment_text']: 
    try:                                                    #If any error comeup I am using exception handling to overcome
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# # printing the first 10 Polarity

# In[15]:


polarity[0:10]


# # defining a feature to store all the polarity values for our df .

# In[16]:


df['polarity']=polarity


# In[17]:


df.head(14)


# # worldcloud of our positive and negative sentences  . Want all the positive sentences in df1 or neagtive sentences in df2

# # for positive sentence the polarity value is 1 and for negative sentence the polarity value is -1 . On the basis of this we will seperate our df1 and df2

# # There are many ways to do this I am using filer method , Defining a filter and applying onto our comments

# In[18]:


df[df['polarity']==1]            #passing the filter in my df ; df['polarity'] is our filter


# # now assigning this to a new df

# In[19]:


comments_positive = df[df['polarity']==1]    


# In[20]:


comments_negative = df[df['polarity']==-1]


# In[21]:


comments_positive.head()


# In[22]:


comments_negative.head()


# # Wordcloud Analysis - Lets suppose to a Data Scientist a task was given for survey of 10k people where he has to analyze what are some of the famous technologies in  particular regions or in a country  and he has stored all the data in some form or database. lets say he has stored the data in the form . Lets say there are various technologies like AI , C , Java , ML ,Dl and he has to analyse what are some of the most famous technologies across the country. So in such case he will create TextBox and inside that we are taking the fonts if the technology and whosoever technology has biigger font is the most popular and rest according to their size . Eg---If DL ha big font it is most popular and in secnd if AI , 3rd Java etc thn it wiil be DL------->AI----------->JAVA----------->ML--------> like that  . This is called wordcloud Analysis

# # to peerform the wordclod analysis typically we need all the data in the form of string nature  with respect to technology use case. It means I have to store all the technologies to the string nature and just pass this string to th wordclod and say to generate a wordcloud for us

# In[23]:


get_ipython().system('pip install wordcloud')


# In[24]:


from wordcloud import WordCloud , STOPWORDS


# In[25]:


comments_negative['comment_text']       #it's a series data structure - Typically 1_d datastructure of our pandas module


# # changing this series data to string . will do with the help of join() inbuilt function  

# In[26]:


total_commentsNeg=' '.join(comments_negative['comment_text'])         #the ' ' firrst is seperator and storing into a new var


# In[27]:


total_commentsNeg[0:100]


# # passing into the wordcloud

# # its for the negative comments

# In[28]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_commentsNeg)
plt.figure(figsize =(15,5))
plt.imshow(wordcloud)
plt.axis('off')

#Here in the string is , he , the , him are stopword so we have to exclude bcoz it doesn;t make any sense in our analysis


# # for positive comments

# In[29]:


total_commentsPos=' '.join(comments_positive['comment_text'])
total_commentsPos[0:100]


# In[30]:


wordcloud = WordCloud(stopwords=set(STOPWORDS)).generate(total_commentsPos)
plt.figure(figsize =(15,5))
plt.imshow(wordcloud)
plt.axis('off')


# # ..................................Emoji's  Analysis...............................................

# # Compute what is the count of Happy , laughing ,Sad or many more emoji's mean how many users use these particular emoji's while posting comment on youtube

# In[31]:


get_ipython().system('pip install emoji')


# In[32]:


import emoji


# In[33]:


df.head(14)


# # First Extracting all the emojis from comment_text present in the data

# In[34]:


print('\U0001F600')    #its a unicode with respect to this emoji . There are many unocodes for different types of unicode 


# # Unicode- Its typically a standard that provides a unique number for each and every character doesn't matter what platform , what device, what application and what language I am using .# Basically It's an encoding standard that assigns a code to every character and symbol in each and every language in the entire world

# In[35]:


#Extracting


# In[36]:


comment = 'trending 😉'


# In[37]:


comment   #i have to extract that emoji


# In[38]:


[c for c in comment if c in emoji.EMOJI_DATA]             


# # Looping to get emoji from entire data ...........................        #we took the df the 10k data not the comments_df the 7M data

# In[39]:


emoji_list =[]             #creating an empty list to store the emoji      
for comment in df['comment_text']:
    for char in comment:
        if char in emoji.EMOJI_DATA:
            emoji_list.append(char)


# In[40]:


len(emoji_list)


# In[41]:


emoji_list[0:20]             #slicng 20 emojis


# # Now we have to check the frequency(occurence) for differnert types of emoji .............. eg - {emoji1 : frequency1 , emoji2 : frequency2}  . Need data in the form of dictionary

# # I am using collection module to do it

# In[42]:


from collections import Counter


# In[43]:


Counter(emoji_list)  #we will get a dictinary with key and value


# # Getting TOP 10 emojis using most_common function of the counter

# In[44]:


Counter(emoji_list).most_common(10)


# In[45]:


#For accessing a key
Counter(emoji_list).most_common(10)[0]


# In[46]:


Counter(emoji_list).most_common(10)[0][0] #getting only emoji 1st one


# In[47]:


Counter(emoji_list).most_common(10)[1][0] #getting only 2nd emoji


# # storing key in one list and emoji in another list

# In[48]:


emojis = [Counter(emoji_list).most_common(10)[i][0] for i in range(10)]   #replacing the index(used for emoji) with i and soting in the emojis list


# In[49]:


emojis


# In[50]:


#for frequency
freqs = [Counter(emoji_list).most_common(10)[i][1] for i in range(10)] #storing into freqs list


# In[51]:


freqs


# In[52]:


import plotly.graph_objs as go


# In[53]:


from plotly.offline import iplot


# In[54]:


trace = go.Bar(x = emojis , y = freqs)


# In[55]:


iplot([trace])


# # ...........................Collecting the Entire Data of the Youtube.............................

# # We have the data for various contries like Canada, France, Denmark, India, Japan, Korea, Mexico, Russia, Us

# # We have to store all the countires data in a DataFrame like full_df       . To do it there are lot of ways . First way is by using 1. O.S model - in this We interact with our OS and pass some path into or OS module and access all the files at some particular path and we have ot do some basic itertations and we are done . ........2nd . Glob - used when no of files is very large and in lot of format and no specifiv structure was present then OS model will not be used Glob is used 

# # OS Model

# In[56]:


import os


# In[57]:


path = r'D:\Datasets\Data Analysis Projects\Youtube_project\Youtube_project\additional_data'


# In[58]:


files = os.listdir(path)


# In[59]:


files


# # need only csv - To do I will use for loop and step parameter

# In[60]:


for i in range(1,len(files),2):           #iterating on the length of the files whatever I have and Here 2 is step parameter
    print(i)                              #output is an order that we need to maintain to get our csv files


# In[61]:


#csv                   and  saving into a new file named as files_csv
files_csv = [files[i] for i in range(0,len(files),2)] 


# In[62]:


files_csv


# In[63]:


#json                 and  saving into a new file named as files_json
files_json = [files[i] for i in range(1,len(files),2)] 


# In[64]:


files_json


# # #saving different countries csv files into a particular DataFrame

# In[65]:


full_df = pd.DataFrame()       #empty df in which we will append

for file in files_csv:
    current_df = pd.read_csv(path+'/'+file, encoding = 'iso-8859-1', error_bad_lines = False)
    
    #I took encoding iso beacise japanese data is also present in the csv and its very  much complex
    #Also want to remov ethe countries name from the starting of the csv files
    
    current_df['country'] = file .split('.')[0][0:2] 
    full_df = pd.concat([full_df,current_df])


# In[66]:


full_df.head()


# In[67]:


full_df.shape


# # ................................Analysing the most liked category.....................................

# # but we don't have any column for category . I have a text file in my pc so I am taking that file

# In[68]:


full_df['category_id'].unique()


# In[69]:


#now reading category_file the text doc
pd.read_csv('D:/Datasets/Data Analysis Projects/Youtube_project/Youtube_project/category_file.txt')


# In[70]:


#Upper output looks messy . SO printing clean data by sperating numbers and category name. 
pd.read_csv('D:/Datasets/Data Analysis Projects/Youtube_project/Youtube_project/category_file.txt',sep =':')


# In[71]:


#upper data also not working . category id is missing . so assigning into a df
cat = pd.read_csv('D:/Datasets/Data Analysis Projects/Youtube_project/Youtube_project/category_file.txt',sep =':')


# In[72]:


cat.reset_index(inplace = True)


# In[73]:


print(cat)


# In[74]:


#customizing the column names           1st = Category_id , 2nd = Category_name

cat.columns=['Category_id', 'Category_name']


# In[75]:


cat


# # making category_id our index

# In[76]:


cat.set_index('Category_id',inplace =True)


# In[77]:


cat


# # changing to the dictionary

# In[78]:


dct = cat.to_dict()       #also storing into a new var dct


# In[79]:


dct


# In[80]:


dct['Category_name']


# # mappinng this dictionary on top of our category_id feature of my full_df DataFrame

# In[81]:


full_df['category_id'].map(dct['Category_name'])


# # now storing this (Category_name) into our full df

# In[82]:


full_df['category_name']= full_df['category_id'].map(dct['Category_name']) 


# In[83]:


full_df.columns  #see category_name is present


# In[84]:


full_df.head()


# # now analysing which category has maximum likes or we say distribution of each and every category with respect to likes.

# # using boxplot to analyse this

# In[85]:


plt.figure(figsize=(15,8))
sns.boxplot(x='category_name' , y ='likes' , data = full_df)
plt.xticks(rotation ='vertical')


# In[86]:


# also printing barplot
plt.figure(figsize=(15,8))
sns.barplot(x='category_name' , y ='likes' , data = full_df)
plt.xticks(rotation ='vertical')


# # regplot for likes

# In[87]:


sns.regplot(data=full_df , x ='views' , y ='likes')


# # ......................Analyzing whether Audience is engaged or not.......................

# # In this we can also think of more features like - dislike rate , dislike rate and commnet rate of the video . These three will help to judge whether our audience is engaging or not .

# # Let's say we have 10k users and 2k have liked the video , 0.5k disliked the video ,  and 1k commented on video . So our percentage of liked , disliked and commented will be 20% , 5% and 10% respectively.

# # computing above calculation with respect to or for my data i.e for full_df

# In[88]:


full_df.columns


# # see in comment we have 4 columns i.e views , likes , dislikes and comment_count . on the basis of these 4 columns we will calculate whether audience is engaged or not

# # foe this I am creating 3 more feature , 1 is like rate , 2 is dislike rate , 3 is comment count rate

# In[89]:


#.............................................for likes..............................................................

full_df['like_rate'] = (full_df['likes']/full_df['views'])*100       #arithmetic cal to find the percentage of like out of the total views and 

#storing in a feature or making a new var to store value


# In[90]:


#.......................................... for dislikes................................................

full_df['dislike_rate'] = (full_df['dislikes']/full_df['views'])*100


# In[91]:


#................................................comment count....................................................
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[92]:


full_df['like_rate']


# In[93]:


full_df['dislike_rate']


# In[94]:


full_df['comment_count_rate']


# In[95]:


full_df.head()


# # using boxplot to analyse the like_rate

# In[96]:


plt.figure(figsize=(15,8))
sns.boxplot(x='category_name' , y ='like_rate' , data = full_df)
plt.xticks(rotation ='vertical')


# In[97]:


#for dislike_rate

plt.figure(figsize=(15,8))
sns.boxplot(x='category_name' , y ='dislike_rate' , data = full_df)
plt.xticks(rotation ='vertical')


# In[98]:


#for comment_count_rate i am plotting barplot

plt.figure(figsize=(15,8))
sns.barplot(x='category_name' , y ='comment_count_rate' , data = full_df)
plt.xticks(rotation ='vertical')


# # correlation between views likes and dislikes

# In[99]:



full_df[['views','likes','dislikes']].corr()


# In[100]:


#plotting a heatmap for corelation

sns.heatmap(full_df[['views','likes','dislikes']].corr(),annot = True )   #annot will print corr value


# # ..............................Analyzing trending videos...............................................

# # for this first we need channel name with number of videos . Eg - C1(channel1) with 10k videos , C2 with 8k and C3 with 7k. If we get this kind of data our task is done

# In[102]:


full_df.head(3)       #in this we don't have any feature with total number of videos


# # To compute total number of videos grouping the data considering my channel_title feature . once I will create a ggroup with channel title feature then we will count the total no of videos with respect to channel title . using groupby() method

# In[103]:


full_df.groupby('channel_title')['video_id'].count() #Here we are getting channel name with no of videos


# In[105]:


#sorting the data
full_df.groupby('channel_title')['video_id'].count().sort_values(ascending = False)


# In[106]:


#converting this data in df                           #getting a df
full_df.groupby('channel_title')['video_id'].count().sort_values(ascending = False).to_frame()


# In[107]:


#reset index

full_df.groupby('channel_title')['video_id'].count().sort_values(ascending = False).to_frame().reset_index()


# In[108]:


#manipulating the name of video_id as it is total count of the videos

full_df.groupby('channel_title')['video_id'].count().sort_values(ascending = False).to_frame().reset_index().rename(columns={'video_id':'total_videos'})


# In[109]:


#now storing it
cdf = full_df.groupby('channel_title')['video_id'].count().sort_values(ascending = False).to_frame().reset_index().rename(columns={'video_id':'total_videos'})


# In[110]:


cdf


# In[111]:


#Now visualizing top 20 data


# In[112]:


import plotly.express as px


# In[117]:


px.bar(data_frame = cdf[0:20] , x ='channel_title' , y ='total_videos' )

