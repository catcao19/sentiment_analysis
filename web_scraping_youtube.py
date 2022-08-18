#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import csv


# In[2]:


#resp = requests.get('https://ca.trustpilot.com/review/www.youtube.com')
url = 'https://ca.trustpilot.com/review/www.youtube.com'
def get_soup(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    return soup


# In[3]:


reviewlist = []

def get_reviews(soup):
    reviews = soup.find_all('section', class_ = 'styles_reviewContentwrapper__zH_9M')
    #reviews = soup.find_all('section', class_ = 'styles_reviewsContainer__3_GQw')
    #try:
    for item in reviews:
        review = {
            'companyName': 'Youtube',
            'datePublished': item.find_all('time')[0]['datetime'],
            'ratingValue': item.find('img').get('alt')[6:7],
            'reviewBody': item.find('p', class_ = 'typography_typography__QgicV typography_body__9UBeQ typography_color-black__5LYEn typography_weight-regular__TWEnf typography_fontstyle-normal__kHyN3').text.strip()
        }
        reviewlist.append(review)
    #except:
        #pass


# In[4]:


for x in range(1, 134):
    try: 
        soup = get_soup(f'https://ca.trustpilot.com/review/www.youtube.com?page={x}')
        get_reviews(soup)
    except:
        pass
    #print(len(reviewlist))


# In[5]:


import pandas as pd
df = pd.DataFrame(reviewlist)
df


# In[ ]:





# In[8]:


df.to_csv('youtube.csv', columns = ['companyName', 'datePublished', 'ratingValue', 'reviewBody'], index = False)


# In[ ]:





# In[ ]:





# In[ ]:




