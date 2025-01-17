# Importing necessary libraries
import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Title and description of the Streamlit app
st.title('Music Recommender System')
st.write("Enter a song name to get similar song recommendations based on music content similarity")

# Adding User ID input
# st.header("Step 1:")
# user_id = st.write('Please enter your User ID', '')

# try:
#     user_id = int(user_id)
# except ValueError:
#     st.write.error("Please enter a valid number")

# Dropdown for selecting User ID
# st.subheader("Select User ID")
user_id = st.selectbox(
    'Please log in with your User ID',
    options=[1001, 1002, 1003, 1004, 1005]
)

# Load your preprocessed dataset
df = pd.read_csv('Preprocessed data.csv')  # Preprocessed music data with numerical features

def recommender(song_name, recommendation_set):
    # Find the index of the song using fuzzy matching
    idx = process.extractOne(song_name, recommendation_set['name'])[2]
    print('Song Selected:', recommendation_set['name'][idx], 'Index:', idx)
    print('Searching for recommendations...')
    
    # Determine the cluster of the selected song
    query_cluster = recommendation_set['cluster'][idx]

    # Filter the dataset to include only points from the same cluster
    filtered_data = recommendation_set[recommendation_set['cluster'] == query_cluster]

    # Reset the index of the filtered data for consistency
    filtered_data = filtered_data.reset_index(drop=True)
    
    # Attempt to find the index of the selected song within the filtered dataset
    try:
        new_idx = filtered_data[filtered_data['name'] == recommendation_set['name'][idx]].index[0]
    except IndexError:
        raise IndexError("The selected song is not found within the filtered cluster data.")

    # Find nearest 10 neighbors, let knn decide algo based on data
    knn5 = NearestNeighbors(metric='euclidean', algorithm='auto', n_neighbors=6) 
    knn10 = NearestNeighbors(metric='euclidean', algorithm='auto', n_neighbors=11) # Add 1 to account for the selected song itself
    knn20 = NearestNeighbors(metric='euclidean', algorithm='auto', n_neighbors=21)
    # knn10 = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=10)
    
    model = knn10

    # Prepare features for KNN
    features = filtered_data.select_dtypes(np.number).drop(columns=['year', 'cluster'])
    model.fit(features)

    # Convert the query point to a DataFrame with the same column names as `features`
    query_point_filtered = pd.DataFrame([features.iloc[new_idx]], columns=features.columns)

    # Find the k nearest neighbors within the same cluster
    distances, indices = model.kneighbors(query_point_filtered)

    # Prepare recommendations
    recommendations = []
    for i in indices[0]:
        if i != new_idx:  # Exclude the selected song itself
            recommendations.append({
                'name': filtered_data.iloc[i]['name'],
                'artist': filtered_data.iloc[i]['artist'],
                'tags': filtered_data.iloc[i]['tags']
            })

    rec_df = pd.DataFrame(recommendations)
    return rec_df

# Initialize session state for playlist
if 'playlist' not in st.session_state:
    st.session_state.playlist = pd.DataFrame(columns=['Song', 'Artist', 'Music Genre Tags', 'Original Song'])

# Input field for song name
song_name = st.text_input("Enter a song that you like:")

if song_name:
    table_df = recommender(song_name, df)
    st.write("These are the recommended songs that you may like: \n", table_df.head(10))
   
    # Filter to show only songs 2 to 6 (index 1 to 5)
    filtered_df = table_df.iloc[1:11].reset_index(drop=True)
    
    # Display the filtered table with checkboxes for selection
    st.write("You may select any recommended songs below and click on the 'Add to Playlist' button to create your personal playlist")

    # Display the filtered DataFrame with checkboxes
    selected_indices = []
    for idx, row in filtered_df.iterrows():
        song_name = row.get('name', 'Unknown Song')
        artist_name = row.get('artist', 'Unknown Artist')
        if st.checkbox(f"{song_name} by {artist_name}", key=idx):
            selected_indices.append(idx)

    # Filter selected songs
    selected_songs = filtered_df.loc[selected_indices]

    # If the user clicks the "Add to Playlist" button, show the selected songs
    if st.button('Add to Playlist'):
        if not selected_songs.empty:
            # Add the original song to the playlist DataFrame
            original_song = pd.DataFrame([{
                'Song': df.loc[original_idx, 'name'],
                'Artist': df.loc[original_idx, 'artist'],
                'Music Genre Tags': df.loc[original_idx, 'tags'],
                'Original Song': True
            }])
            
            st.write("Your Playlist")
            st.dataframe(pd.concat([st.session_state.playlist, original_song, selected_songs], ignore_index=True), use_container_width=True, hide_index=True)
            
            # Update session state playlist
            st.session_state.playlist = pd.concat([st.session_state.playlist, original_song, selected_songs], ignore_index=True)
            
            # Save the updated playlist to a CSV file
            if 'user_id' in st.session_state and st.session_state.user_id:
                filename = f'Playlist_{st.session_state.user_id}.csv'
                st.session_state.playlist.to_csv(filename, index=False)
                st.write(f"Playlist saved as {filename}")
            else:
                st.write("User ID is not set. Cannot save playlist.")
        else:
            st.write("No songs selected, please try again.")


# songs = ["Song 1","Song 2","Song3"]
# Create a radio button for each song
# selected_songs = st.radio("Select a song to add to your playlist:", options=songs)

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


# # Adding a sidebar
# st.sidebar.title("Sidebar")
# option = st.sidebar.selectbox(
#     'Please enter your User ID',
#     list(range(1001, 1006)))
# st.image('pic.jpg')
# Age=st.sidebar.radio('Please enter your Age Group',options=['Under 20','20+','30+','40+','Over 50'])


