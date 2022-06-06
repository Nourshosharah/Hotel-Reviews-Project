import streamlit as st
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
from tfidf_and_reccomender_funs import recommend_based_feateuers,filter_by_name,get_value

data_recomm_guests=pd.read_pickle("Data/df_guests_all.pkl")
data_filters=pd.read_pickle("Data/data_filters.pkl")


with open('Data/HOTEL_NAME.npy', 'rb') as f2:
    hotels = np.load(f2,allow_pickle=True)

with open('Data/test.npy', 'rb') as f:
    family = np.load(f,allow_pickle=True)
    Couple =  np.load(f,allow_pickle=True)
    Group =  np.load(f,allow_pickle=True)
    Solo =  np.load(f,allow_pickle=True)
    friends =  np.load(f,allow_pickle=True)

with open('Data/guests.npy', 'rb') as f2:
    guests = np.load(f2,allow_pickle=True)

with open('Data/Reviewer_Nationality.npy', 'rb') as f2:
    Reviewer_Nationality = np.load(f2,allow_pickle=True)



with open('Data/country.npy', 'rb') as f2:
    country = np.load(f2,allow_pickle=True)






app_mode = st.sidebar.selectbox('Select Page',['The_Best_For_You','guests_talks'])
if app_mode=='The_Best_For_You':   
    
    image = Image.open("Data/Cover.jpg")
    st.image(image)
 
    st.title('Recommended by guest_featuers')


    st.subheader('Sir/Mme , YOU need to fill all neccesary informations in order to get a reply to your best stay!')
    st.sidebar.header("Informations about the client :")
    trip_type_dict = {'Leisure trip':"Leisure trip", 'Business trip':"Business trip" }

    from_where=st.sidebar.selectbox('from',Reviewer_Nationality)
    to_where=st.sidebar.selectbox('to',country)

    trip_type=st.sidebar.radio('trip_type',tuple(trip_type_dict.keys()))


    guests=st.sidebar.selectbox('guests',guests)
    if guests =="Family":
        rooms_type=st.sidebar.selectbox('rooms_type',family)
    if guests =="Couple":
        rooms_type=st.sidebar.selectbox('rooms_type',Couple)
    if guests =="Group":
        rooms_type=st.sidebar.selectbox('rooms_type',Group)
    if guests =="Travelers with friends":
        rooms_type=st.sidebar.selectbox('rooms_type',friends)
    if guests =="Solo traveler":
        rooms_type=st.sidebar.selectbox('rooms_type',Solo)


    stayed=st.sidebar.slider('stayed',0,31,0,)
    tags_for_custumer=guests+ ' '+ trip_type+" "+rooms_type+" "+str(stayed)


    if st.button("show me best Recommendations"):
      
        data_recommendations=recommend_based_feateuers(data_recomm_guests,tags_for_custumer,from_where,to_where)
        st.table (data_recommendations)


elif app_mode =='guests_talks':
    image = Image.open("Data/Cover_1.jpg")
    st.image(image)

    st.title('fiter by hotel name')
    st.write('please write Hotel name to show you best reviews')
    Hotel_Name=st.sidebar.multiselect('Hotel_Name',hotels)
    print("Hotel_Name",str(Hotel_Name))


    if st.button("show me best reviews"):

        data_filters=filter_by_name(data_filters,Hotel_Name)
        st.table (data_filters)




