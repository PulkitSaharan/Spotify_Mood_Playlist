import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from PIL import Image
import joblib
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from random import sample


pipeline = joblib.load('./song-cluster-model.joblib')
cid = 'masked'
secret = 'masked'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

df = pd.read_csv("spotify_songs.csv")
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df['Labels']=pipeline.predict(df[audio_features])
#Create header
st.write("""# Spotify Mood Playlist""")
st.write("""## How it works""")
st.write("Get a mood playlist with songs similar to your favorite song. Just provide the csong input."
         "Alternatively, play with the mood sliders to get a desired playlist.")


#image
image = Image.open('spotify.jpeg')
st.image(image)

#Bring in the data
data = pd.read_csv('spotify_songs.csv')
st.write("## THE DATA BEING USED")
data

#Create and name sidebar
st.sidebar.header('Choose Your Playlist Preferences')
artist_name = st.sidebar.text_input("Artist Name")
track_name = st.sidebar.text_input("Track Name")

attr_check_box = st.sidebar.checkbox("Provide audio features instead of artist and song name?")
st.sidebar.checkbox("Do you want less popular songs?")
st.sidebar.write("Click Moodify multiple times to get different playlists")
submit_button = st.sidebar.button('Moodify!')

def user_input_features():

    if attr_check_box:
        danceability = st.sidebar.slider('Danceability', 0.0, 1.0, 0.5, 0.01)
        energy = st.sidebar.slider('Energy', 0.0, 1.0, 0.5, 0.01)
        key = st.sidebar.slider('Key', 0, 11, 10, 1)
        loudness = st.sidebar.slider('Loudness', -46.0, 2.0, 0.5, 0.01)
        mode = st.sidebar.slider('Mode', 0, 1, 0, 1)
        speechiness = st.sidebar.slider('Speechiness', 0.0, 1.0, 0.5, 0.01)
        acousticness = st.sidebar.slider('Acousticness', 0.0, 1.0, 0.5, 0.01)
        instrumentalness = st.sidebar.slider('Instrumentalness', 0.0, 1.0, 0.5, 0.01)
        liveness = st.sidebar.slider('Liveness', 0.0, 1.0, 0.5, 0.01)
        valence = st.sidebar.slider('Valence', 0.0, 1.0, 0.5, 0.01)
        tempo = st.sidebar.slider('Tempo', 0.0, 240.0, 20.0, 0.1)
        


        user_data = {'danceability': danceability,
                 'energy': energy,
                 'key': key,
                 'loudness': loudness,
                 'mode': mode,
                 'speechiness': speechiness,
                 'acousticness': acousticness,
                 'instrumentalness': instrumentalness,
                 'liveness': liveness,
                 'valence': valence,
                 'tempo': tempo}
        features = pd.DataFrame(user_data, index=[0])
        features = features[audio_features]
        return features
    elif artist_name and track_name:
        song = sp.search(q="artist:" + artist_name + " track:" + track_name, type="track")
        song_id = song['tracks']['items'][0]['id']
        features = pd.DataFrame(sp.audio_features(song_id))
        features = features[audio_features]
        return features

def create_playlist(new_song, df, label, threshold = 0.1):

    cols = ["track_name", "track_artist", "track_album_name","track_popularity", "playlist_subgenre"]
    num_songs = 15
    song_label=pipeline.predict(new_song)[0]
    cluster_df=df[df['Labels']==song_label]
    song_dist=[]
    
    for i in range(len(cluster_df)):
        song_dist.append(distance.cdist(np.array(new_song.head(1)), np.array(cluster_df[cluster_df['Labels']==label].loc[:,cluster_df.columns.isin(audio_features)][i:i+1]), metric = 'minkowski', p = 2)[0])
      
    cluster_df['song_dist']=song_dist
    cluster_df['normal_dist']=MinMaxScaler().fit_transform(cluster_df[['song_dist']])
    #playlist=list(cluster_df[cluster_df['normal_dist']<=threshold]['track_name'])
    playlist=cluster_df[cluster_df['normal_dist']<=threshold][cols]
    final_playlist=playlist.sample(min(len(playlist),num_songs)).reset_index(drop = True)
    final_playlist.columns = ["Track Name", "Artist", "Album","Track Popularity", "Genre"]
    #final_playlist = pd.DataFrame(final_playlist, columns = ["Song Name"], index = range(1,num_songs+1))
    return(final_playlist)

def generate_playlist(df_user):
    cluster_num = predict_cluster(df_user)
    st.write(cluster_num)
    playlist = create_playlist(df_user, df, cluster_num, 0.1)
    playlist
    

def predict_cluster(df_user):
    cluster_num = pipeline.predict(df_user)
    return cluster_num[0]

df_user = user_input_features()


st.write("## YOUR CHOSEN MOOD ATTRIBUTES: ")
df_user

if submit_button:
    #df_user = user_input_features()

    st.write("## GENERATED PLAYLIST")
    generate_playlist(df_user)



