#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


# In[3]:


df_listings = pd.read_csv(r'C:\Users\Melikzade\Desktop\SQL Course Materials\listings.csv')
df_calendar = pd.read_csv(r'C:\Users\Melikzade\Desktop\SQL Course Materials\calendar.csv')
df_reviews = pd.read_csv(r'C:\Users\Melikzade\Desktop\SQL Course Materials\reviews.csv')


# In[4]:


df_listings.head()


# In[5]:


df_reviews.head()


# In[6]:


df_listings.describe()


# In[7]:


df_calendar.head()


# In[8]:


# Convert data types of df_calendar columns to reduce memory usage
df_calendar['listing_id'] = df_calendar['listing_id'].astype('int32')
df_calendar['date'] = pd.to_datetime(df_calendar['date'])
df_calendar['available'] = df_calendar['available'].map({'t': True, 'f': False})
# Remove dollar signs and commas from price column and convert to float
df_calendar['price'] = df_calendar['price'].replace('[\$,]', '', regex=True).astype(float)

# Print adjusted data types
print(df_calendar.dtypes)


# In[9]:


# Convert data types of df_reviews columns to reduce memory usage
df_reviews['listing_id'] = df_reviews['listing_id'].astype('int32')
df_reviews['id'] = df_reviews['id'].astype('int32')
df_reviews['date'] = pd.to_datetime(df_reviews['date'])
df_reviews['reviewer_id'] = df_reviews['reviewer_id'].astype('int32')
df_reviews['reviewer_name'] = df_reviews['reviewer_name'].astype('category')
df_reviews['comments'] = df_reviews['comments'].astype('string')

# Print adjusted data types
print(df_reviews.dtypes)


# In[10]:


# Rename id column to listing_id
df_listings = df_listings.rename(columns={'id': 'listing_id'})

# Verify that the column name has been changed
print(df_listings.columns)


# In[11]:


# Group availability by date and count the number of available and unavailable days
availability_counts = df_calendar.groupby(['date', 'available']).size().unstack(fill_value=0)

# Plotting available and unavailable time throughout the year
plt.figure(figsize=(12, 6))
availability_counts.plot(kind='line', stacked=True)
plt.title('Availability of Airbnb Listings Throughout the Year')
plt.xlabel('Date')
plt.ylabel('Number of Listings')
plt.grid(True)
plt.show()


# In[17]:


# Group by date and calculate the average price per day
average_price_per_day = df_calendar.groupby('date')['price'].mean()

# Plotting average price per day
plt.figure(figsize=(12, 6))
average_price_per_day.plot(color='blue')

plt.title('Average Price per Day Throughout the Year')
plt.xlabel('Date')
plt.ylabel('Average Price')
plt.grid(True)
plt.show()


# In[21]:


# Merge df_listings and df_reviews on listing_id
merged_df = pd.merge(df_listings, df_reviews, on='listing_id')

# Filter data for Seattle
seattle_data = merged_df[merged_df['city'] == 'Seattle']

# Group reviews by listing_id and count the number of reviews
reviews_counts = seattle_data.groupby('listing_id')['id'].count()

# Sort reviews by number of reviews and get the most and least reviewed listings
most_reviewed_listings = reviews_counts.sort_values(ascending=False).head(10)
least_reviewed_listings = reviews_counts.sort_values().head(10)

# Plotting most reviewed listings
plt.figure(figsize=(12, 6))
most_reviewed_listings.plot(kind='barh')
plt.title('Most Reviewed Seattle Airbnb Listings')
plt.xlabel('Number of Reviews')
plt.ylabel('Listing ID')
plt.grid(True)
plt.show()

# Plotting least reviewed listings
plt.figure(figsize=(12, 6))
least_reviewed_listings.plot(kind='barh')
plt.title('Least Reviewed Seattle Airbnb Listings')
plt.xlabel('Number of Reviews')
plt.ylabel('Listing ID')
plt.grid(True)
plt.show()


# In[23]:


# Group the listings by neighborhood
neighborhood_groups = df.groupby('neighborhood')

# Iterate over each neighborhood group
for neighborhood, group_data in neighborhood_groups:
    print(f"Neighborhood: {neighborhood}")
    print("Number of Listings:", len(group_data))
    print("Sample Descriptions:")
    for revı in group_data['description'].head(3):
        print("-", description)
    print("\n")


# In[42]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# Preprocess the listing descriptions
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    filtered_text = [lemmatizer.lemmatize(word) for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

df_listings['description'] = df_listings['description'].apply(preprocess_text)

# Segment the listings by neighborhood
neighborhoods = df_listings['neighbourhood'].unique()

# Analyze the text data for each neighborhood
for neighborhood in neighborhoods:
    listings = df_listings[df_listings['neighbourhood'] == neighborhood]
    descriptions = listings['description'].tolist()
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Perform topic modeling using NMF
    nmf = NMF(n_components=10)
    nmf.fit(tfidf_matrix)
    
    # Display the top 10 words for each topic
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(nmf.components_):
        print(f"Neighborhood: {neighborhood}")
        print(f"Topic {topic_idx + 1}:")
        top_words_idx = topic.argsort()[-10:][::-1] 
        top_words = [feature_names[i] for i in top_words_idx]
        print(top_words)
        
        # Generate a word cloud for the current topic
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(top_words))
        
        # Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Neighborhood: {neighborhood} - Topic {topic_idx + 1}")
        plt.axis('off')
        plt.show()


# In[ ]:





# In[ ]:




