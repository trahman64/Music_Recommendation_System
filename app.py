
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Music Recommendation System", layout="wide")

st.title("Music Recommendation System")

st.write(
    "Explore how danceability and energy vary across musical genres. "
    "Select genres to compare their audio characteristics."
)

# Load and clean the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/spotify_dataset.csv")

    df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna(subset=["track_name", "artists", "album_name"])
    df = df.drop_duplicates()

    return df

music_df = load_data()

# Allow users to select genres
all_genres = sorted(music_df["track_genre"].unique())

default_genres = [
    "pop", "rock", "classical", "hip-hop",
    "jazz", "electronic", "acoustic"
]

selected_genres = st.multiselect(
    "Choose genres to compare:",
    options=all_genres,
    default=[g for g in default_genres if g in all_genres]
)

if selected_genres:
    # Filter the dataset
    filtered_df = music_df[
        music_df["track_genre"].isin(selected_genres)
    ]

    # Calculate genre averages
    genre_summary = (
        filtered_df.groupby("track_genre")[
            ["danceability", "energy"]
        ].mean().reset_index()
    )

    # Interactive visualization
    fig = px.scatter(
        genre_summary,
        x="danceability",
        y="energy",
        color="track_genre",
        text="track_genre",
        title="Average Danceability vs. Energy by Genre",
        labels={
            "danceability": "Average Danceability",
            "energy": "Average Energy",
            "track_genre": "Genre"
        }
    )

    fig.update_traces(
        marker=dict(size=15),
        textposition="top center"
    )

    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Genre Summary Statistics")
    st.dataframe(genre_summary.round(3), hide_index=True)

else:
    st.info("Select at least one genre to display the comparison.")
