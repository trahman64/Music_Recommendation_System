
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommendation System", layout="wide")

st.title("Music Recommendation System")

st.write(
    "Explore how audio characteristics vary across musical genres "
    "and discover songs with similar musical features."
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


# Prepare the recommendation dataset
@st.cache_data
def prepare_recommendations(df):

    recommend_music_df = df[df["tempo"] > 0].copy()

    recommend_music_df = recommend_music_df.drop_duplicates(
        subset=["track_name", "artists"]
    ).copy()

    recommend_music_df = recommend_music_df.reset_index(drop=True)

    recommend_music_df["song_label"] = (
        recommend_music_df["track_name"]
        + " - "
        + recommend_music_df["artists"]
    )

    audio_features = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "instrumentalness",
        "speechiness",
        "liveness",
        "loudness",
        "tempo"
    ]

    scaler = StandardScaler()

    scaled_features = scaler.fit_transform(
        recommend_music_df[audio_features]
    )

    return recommend_music_df, scaled_features


recommend_music_df, scaled_features = prepare_recommendations(music_df)


# Recommendation function
def recommend_songs(song_index, n=4):

    selected_song = scaled_features[song_index].reshape(1, -1)

    similarities = cosine_similarity(
        selected_song, scaled_features
    )[0]

    sorted_indices = np.argsort(similarities)[::-1]

    sorted_indices = sorted_indices[
        sorted_indices != song_index
    ]

    top_indices = sorted_indices[:n]

    results = recommend_music_df.iloc[top_indices][
        ["track_name", "artists", "track_genre"]
    ].copy()

    results["similarity"] = similarities[top_indices]

    return results


# Create tabs
tab1, tab2 = st.tabs([
    "Explore Music",
    "Get Recommendations"
])


# TAB 1: Genre exploration
with tab1:

    st.subheader("Explore Music Genres")

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

        eda_tab1, eda_tab2 = st.tabs([
        "Genre Averages",
        "Danceability Distribution"])
        filtered_df = music_df[
            music_df["track_genre"].isin(selected_genres)
        ]

        genre_summary = (
            filtered_df.groupby("track_genre")[
                ["danceability", "energy"]
            ].mean().reset_index()
        )

        with eda_tab1:
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
                textposition="top center")
            fig.update_layout(showlegend=False)

            
            fig.update_xaxes(
            tickfont=dict(size=14, color="#333333"),
            title_font=dict(size=16, color="#333333"),
            range=[0, 1])

            fig.update_yaxes(
            tickfont=dict(size=14, color="#333333"),
            title_font=dict(size=16, color="#333333"),
            range=[0, 1])
            fig.update_layout(
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_color="black"
            ),showlegend=False)
            

            st.plotly_chart(fig, use_container_width=True)
            most_danceable = genre_summary.loc[
            genre_summary["danceability"].idxmax()]

            most_energetic = genre_summary.loc[genre_summary["energy"].idxmax()]

            st.success(
                f"Among the selected genres, {most_danceable['track_genre']} "
                f"has the highest average danceability "
                f"({most_danceable['danceability']:.3f}), while "
                f"{most_energetic['track_genre']} has the highest average "
                f"energy ({most_energetic['energy']:.3f})."
            )

            with st.expander("View Genre Summary Statistics"):

                st.dataframe(
                    genre_summary.round(3),
                    hide_index=True)
        # EDA 2: Danceability distribution by genre

        with eda_tab2:
            st.subheader("Danceability Distribution Across Genres")

            fig_box = px.box(
                filtered_df,
                x="track_genre",
                y="danceability",
                color="track_genre",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={
                    "track_genre": "Genre",
                    "danceability": "Danceability"
                }
            )
            fig_box.update_traces(fillcolor=None, opacity=0.85)
            fig_box.update_xaxes(
            tickfont=dict(size=14, color="#333333"),
            title_font=dict(size=16, color="#333333"))

            fig_box.update_yaxes(
            tickfont=dict(size=14, color="#333333"),
            title_font=dict(size=16, color="#333333"),
            range=[0, 1])
            fig_box.update_layout(showlegend=False)
            fig_box.update_layout(hoverlabel=dict(bgcolor="white",font_size=14,font_color="black"),
                                showlegend=False)

            st.plotly_chart(fig_box, use_container_width=True)

            danceability_spread = (filtered_df.groupby("track_genre")["danceability"].std())

            most_variable_genre = danceability_spread.idxmax()

            st.info(
                f"Among the selected genres, {most_variable_genre} "
                f"has the greatest variation in danceability "
                f"(standard deviation = "
                f"{danceability_spread.max():.3f})."
            )
            

    else:
        st.info("Select at least one genre to display the comparison.")


# TAB 2: Music recommendations
with tab2:
    st.subheader("Find Similar Songs")

    st.write(
        "Choose a song to discover other songs "
        "with similar audio characteristics."
    )

    selected_song = st.selectbox(
        "Search for a song:",
        recommend_music_df["song_label"]
    )

    song_index = recommend_music_df[
        recommend_music_df["song_label"] == selected_song
    ].index[0]

    selected_song_info = recommend_music_df.iloc[song_index]

    st.write("**Selected Song**")
    st.write(f"**Title:** {selected_song_info['track_name']}")
    st.write(f"**Artist:** {selected_song_info['artists']}")
    st.write(f"**Genre:** {selected_song_info['track_genre']}")

    num_recommendations = st.slider(
    "Number of recommendations:",
    min_value=1,
    max_value=10,
    value=5
)

    if st.button("Recommend Songs"):
         st.session_state["show_recommendations"] = True

    if st.session_state.get("show_recommendations", False):
        
        recommendations = recommend_songs(song_index, num_recommendations)

        st.subheader("Your Recommendations")

        display_recommendations = recommendations.rename(
        columns={
            "track_name": "Song",
            "artists": "Artist",
            "track_genre": "Genre",
            "similarity": "Cosine Similarity"
        })

        st.dataframe(
            display_recommendations,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Cosine Similarity": st.column_config.NumberColumn(
                    "Cosine Similarity",
                    format="%.3f"
                )
            }
        )
        st.subheader("Why Was This Song Recommended?")

        top_song_index = recommendations.index[0]

        comparison_options = (
        recommendations["track_name"]
        + " - "
        + recommendations["artists"])

        selected_comparison = st.selectbox(
            "Choose a recommended song to compare:",
            comparison_options)

        comparison_index = comparison_options[
            comparison_options == selected_comparison
        ].index[0]

        top_song = recommendations.loc[comparison_index]

        features_to_compare = [
            "danceability",
            "energy",
            "valence",
            "acousticness",
            "instrumentalness"
        ]

        selected_values = recommend_music_df.loc[
            song_index, features_to_compare
        ].to_numpy(dtype=float)

        top_song_index = recommend_music_df[
            (recommend_music_df["track_name"] == top_song["track_name"]) &
            (recommend_music_df["artists"] == top_song["artists"])
        ].index[0]

        recommended_values = recommend_music_df.loc[
            top_song_index, features_to_compare
        ].to_numpy(dtype=float)

        comparison_df = pd.DataFrame({
            "Feature": features_to_compare,
            "Selected Song": selected_values,
            "Recommended Song": recommended_values
        })

        comparison_df = comparison_df.melt(
            id_vars="Feature",
            var_name="Song",
            value_name="Value"
        )

        fig_comparison = px.bar(
            comparison_df,
            x="Feature",
            y="Value",
            color="Song",
            barmode="group",
            title="Audio Feature Comparison",
            labels={"Value": "Feature Value"}
        )

        fig_comparison.update_yaxes(range=[0, 1])

        fig_comparison.update_layout(
        hoverlabel=dict(
            bgcolor="#2C3E50",
            font_color="white",
            font_size=14
        ),
        legend=dict(
            font=dict(size=14, color="#333333")
        ))

        fig_comparison.update_xaxes(
            tickfont=dict(size=14, color="#333333"),
            title_font=dict(size=16, color="#333333")
        )

        fig_comparison.update_yaxes(
            tickfont=dict(size=14, color="#333333"),
            title_font=dict(size=16, color="#333333"),
            range=[0, 1]
        )
        st.plotly_chart(fig_comparison, use_container_width=True)