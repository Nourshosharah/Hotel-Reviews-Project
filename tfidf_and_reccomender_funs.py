import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from PIL import Image
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import config
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st



data_recomm_guests=pd.read_pickle("Data/df_guests_all.pkl")

data_filters=pd.read_pickle("Data/data_filters.pkl")


def get_recommendations(N, scores):
    # load in recipe dataset 
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:20]
    print(top)
    # create dataframe to load in recommendations 
    recommendation = pd.DataFrame(columns = ['Hotel_Address', 'Hotel_Name', 'Average_Score','lat', 'lng'])
    count = 0
    for i in top:
        recommendation.at[count, 'Hotel_Address'] = data_recomm_guests['Hotel_Address'][i]
        recommendation.at[count, 'Hotel_Name'] = data_recomm_guests['Hotel_Name'][i]
        recommendation.at[count, 'Average_Score'] = data_recomm_guests['Average_Score'][i]
        recommendation.at[count, 'lat'] = data_recomm_guests['lat'][i]
        recommendation.at[count, 'lng'] = data_recomm_guests['lng'][i]
        recommendation.at[count, 'score'] = "{:.3f}".format(float(scores[i]))
        count += 1
        df2=recommendation.drop_duplicates( subset=['Hotel_Name'],keep='last')
        df2=df2[['Hotel_Address','Hotel_Name','lat', 'lng','Average_Score']]
        df_sort=df2.sort_values(by='Average_Score',ascending=False)[:20]
    return df_sort




def recommend_based_feateuers(data_recomm_guests,tags_for_custumer,from_where,to_where,N=5):
        # TF-IDF feature extractor 
    data_recomm_guests=data_recomm_guests[(data_recomm_guests["country"]==to_where)&(data_recomm_guests["Reviewer_Nationality"]==from_where)]
    data_recomm_guests=data_recomm_guests.reset_index()
    
    data_recomm_guests['all'] = data_recomm_guests["all"].values.astype('U')
    # TF-IDF feature extractor 
    tfidf = TfidfVectorizer()
    tfidf.fit(data_recomm_guests['all'])
    tfidf_recipe = tfidf.transform(data_recomm_guests['all'])
    # save the tfidf model and encodings 
    with open(config.TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(tfidf, f)
    with open(config.TFIDF_ENCODING_PATH, "wb") as f:
        pickle.dump(tfidf_recipe, f)

    # # load in tdidf model and encodings 
    with open(config.TFIDF_ENCODING_PATH, 'rb') as f:
        tfidf_encodings = pickle.load(f)
    with open(config.TFIDF_MODEL_PATH, "rb") as f:
        tfidf = pickle.load(f)
    ingredients_tfidf = tfidf.transform([tags_for_custumer])
    # calculate cosine similarity between actual recipe ingreds and test ingreds
    cos_sim = map(lambda x: cosine_similarity(ingredients_tfidf, x), tfidf_encodings)
    scores = list(cos_sim)
    # Filter top N recommendations 
    recommendations = get_recommendations(N, scores)
    return recommendations


@st.cache(suppress_st_warning=True)
def filter_by_name(data,hotel_name):
  new_data=data[data["Hotel_Name"]==hotel_name[0]][["Negative_Review","Positive_Review"]]
  all_neg_words=new_data['Negative_Review'].values
  all_pos_words=new_data["Positive_Review"].values

  cv = CountVectorizer(analyzer = 'word',stop_words = 'english',max_features = 50,ngram_range=(3,3))
  most_negative_words = cv.fit_transform(all_neg_words)
  neg_counts = most_negative_words.sum(axis=0)
  neg_counts = cv.vocabulary_ 

  most_positive_words = cv.fit_transform(all_pos_words)
  pos_counts = most_positive_words.sum(axis=0)
  pos_counts = cv.vocabulary_

  negative_positive_words=pd.DataFrame()
  negative_positive_words["negative_sentence"]=neg_counts.keys()
  negative_positive_words["positive_sentence"]=pos_counts.keys()
  negative_positive_words["frequency_neg"]=neg_counts.values()
  negative_positive_words["frequency_pos"]=pos_counts.values()
  final_data=negative_positive_words.sort_values(by=["frequency_neg","frequency_pos"],ascending=[False,False]).reset_index()[:40]
  return final_data[["negative_sentence","positive_sentence"]]


@st.cache(suppress_st_warning=True)
def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val == key:
            return value