# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name and get 5 similar song recommendations based on content similarity")

# Adding a sidebar
st.sidebar.title("Sidebar")
option = st.sidebar.selectbox(
    'Please enter your User ID',
    list(range(1001, 1006)))
   
# st.image('pic.jpg')
# Age=st.sidebar.radio('Please enter your Age Group',options=['Under 20','20+','30+','40+','Over 50'])

# Load your preprocessed dataset
df = pd.read_csv('Preprocessed data.csv')  # Preprocessed music data with numerical features

# Define the Recommender class
class Recommender:
    def __init__(self):
        self.knn = NearestNeighbors(n_neighbors=11, metric='euclidean')
    
    def fit(self, data):
        self.data = data
        self.features = data.select_dtypes(np.number).drop(columns=['year', 'cluster'])
        self.knn.fit(self.features)
    
    def recommender(self, song_name, recommendation_set):
        idx = process.extractOne(song_name, recommendation_set['name'])[2]
        print('Song Selected:', recommendation_set['name'][idx], 'Index:', idx)
        print('Searching for recommendations...')
        
        query_cluster = recommendation_set['cluster'][idx]
        filtered_data = recommendation_set[recommendation_set['cluster'] == query_cluster]
        filtered_data = filtered_data.reset_index(drop=True)
        
        try:
            new_idx = filtered_data[filtered_data['name'] == recommendation_set['name'][idx]].index[0]
        except IndexError:
            raise IndexError("The selected song is not found within the filtered cluster data.")

        query_point_filtered = pd.DataFrame([self.features.iloc[new_idx]], columns=self.features.columns)
        distances, indices = self.knn.kneighbors(query_point_filtered)

        recommendations = []
        for i in indices[0]:
            if i != new_idx:
                recommendations.append({
                    'name': filtered_data.iloc[i]['name'],
                    'artist': filtered_data.iloc[i]['artist'],
                    'tags': filtered_data.iloc[i]['tags']
                })

        return pd.DataFrame(recommendations)

# Load the trained KNN model from the pickle file
with open('knn_model.pkl', 'rb') as f:
    spotify = pickle.load(f)

# Input field for song name
song_name = st.text_input("Enter a song that you like:")

# If the user has entered a song name, perform the recommendation
if song_name:
    recommended_songs = spotify.recommender(song_name)
    st.write("\n", recommended_songs.head(10))

songs = ["Song 1","Song 2","Song 3","Song 4","Song 5","Song 6","Song 7","Song 8","Song 9","Song 10"]

# Create a radio button for each song
selected_songs = st.radio("Select a song to add to your playlist:", options=songs)

# Button to submit the selection
if st.button("Add to Playlist"):
    # Here you can store the selected song as needed
    st.write(f"You have added '{selected_songs}' to your playlist!")

# Adding a button
if st.button('Click to refresh playlist'):
    st.write('Updated Playlist')

st.write("Here is your current playlist:")
# Dataframe display
playlist_df = pd.DataFrame({
    'Songs': ["Moonlight Sonata", "Viva la Vida", "Toccata and Fugue in D Minor"],
    'Artist': ["Ludwig van Beethoven", "Coldplay", "Johann Sebastian Bach"]
    })
st.write(playlist_df)

# Create a slider
# rating = st.slider("Please rate the recommended song (5 being Highest", min_value=1, max_value=5, value=1)

# Display the selected value
# st.write("You have given a rating of ", rating)
