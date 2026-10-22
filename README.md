# Music Recommendation System
## Abstract
This project examines how audio features vary across various music genres, and how can these features be used to recommend similar songs. Initially, two exploratory data analysis (EDA) have been carried out in a notebook called music_analysis_EDA to see if there is any patterns in audio features such as in danceability and energy across genres. After that, a content based recommendation system is developed to classify similar songs using standardized audio features and cosine similarity. Lastly, an interactive streamlit application is launched where users can visualize both of the EDA as well as get personalized recommendations of the song based on their selection.
## Research Question
How do audio characteristics vary across musical genres, and how can these features be used to generate content-based music recommendations?
## Target Audience
Audience for this project is any music listener who wants to explore different music genre and different songs with similar audio characteristics.
## Dataset
### Dataset Description
The dataset used in the project is from the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) by Maharshi Pandya from Kaggle. There are 114000 tracks across 125 music genres where each observations includes information about song titles, artists, popularity and other audio characteristics. Dataset can be accessed under the directory data.
### Data Cleaning
Dataset had to be cleaned before any analysis was done. At first, unnecessary column was removed which was the column 'unnamed'. Secondly, there was one observation with missing song titles, artist names and album names which was also removed. Thirdly, exact duplicated rows were removed. For the recommendation system, songs with the same title and artist were kept only once to avoid recommending duplicate songs, and also songs with zero tempo were removed because there is a high chance they may indicate unavailable tempo information. These information were kept for EDA. 
## Exploratory Data Analysis
EDA was done to get a better understanding of audio characteristics differ across music genres. For the initial analysis, seven genres were selected which includes rock, pop, classical, hip-hop, jazz, electronic and acoustic.
### EDA 1: Average Danceability vs Avergage Energy by Genre
To do this, a scatter plot was created using plotly. This showed us that hip-hop has the highest average danceability while electronic music has the highest average energy. On the other hand, classical music has the lowest average danceability and energy. However, rock music has relatively high energy but it has lower danceability in contrast with hip-hop and electronic music. Exact numbers can be found in the music_analysis_EDA.ipynb.
### EDA 2: Danceability Distribution across Genre
To do this, a box plot was done which displayed how danceability varies within and across genres. This portrayed that hip-hop has the highest median danceability where as classical music has the lowest. Classical music has a low median but it also has a high variation. In general, pop and electronic have higher danceability values.

Overall, both EDA suggests that audio features do differ across music genres even though some might have a bit overlap. Therefore, it makes sense to use audio features to compare songs instead of using genres for the recommendation system.

## Recommendation System
A content-based recommendation system was developed to identify songs with similar audio characteristics. This recommendation system did not take account of any user listening history or genre. To develop this, nine audio features were selected which includes danceability, energy, valence, acousticness, instrumentalness, speechiness, liveness, loudness and tempo. Since each of them do not have the same scale, StandardScaler was applied so that every value is standardized, and no feature dominates in the similarity calculation. After that, cosine similarity was used to measure how similar each song is to the selected song based on its standardized audio features. This shows the songs with highest similarity ranked from the top, and it does not include the selected song itself.
## Streamlit Application
An interactive web application was developed using Streamlit so that users can access the analysis and the recommendation system. 
### Explore Music Genres
This tab includes the two EDA analysis discussed above which are again split into two tabs. One flexibility is that user can select the music genres they want to explore, and the visualization and the statistics will be updated dynamically. Initially by default, 7 genres will be shown like it was done in the music_analysis_EDA.ipynb but they can update it. First visualization shows a scatterplot between average danceability and energy, and second visualization shows a box plot of danceability distribution across genres. A summary statistics table is also there for the scatter plot between average danceability and energy.
### Get Recommendations
For this tab, users can search and select a song. They also have the flexibility to choose the number of recommendations up to 10 using the slider. After that, a list of similar songs containing similar audio features will be visible. Moreover, if user wants, they can also see why that particular song was recommented using the bar chart. The app can be accessed with from [here](https://musicrecommendationsystem-84ueyyfzzc5gp5eamvmxza.streamlit.app/).
## How to run the project
The steps to run the music recommendation system locally will be described below.
### Cloning the Repository
The repository can be cloned from https://github.com/trahman64/Music_Recommendation_System.git
The particular commands are:
git clone https://github.com/trahman64/Music_Recommendation_System.git
cd Music_Recommendation_System

### Dependencies
Any libraries required for this project is outlined inside requirements.txt. Run the command below to get everything installed.
pip install -r requirements.txt

### Dataset
Data file is put inside the data directory.

### Running the Streamlit Application
Application can be run with the following command locally:
streamlit run app.py
A url for the deployed application is [here](https://musicrecommendationsystem-84ueyyfzzc5gp5eamvmxza.streamlit.app/).

### Running the Jupyter Notebooks
music_analysis_EDA.ipynb contains data cleaning, exploratory analysis and visualizations.
Recommendation_algorithm.ipynb contains feature preprocessing and content-based recommendation system. 
## Limitation
There are some limitations to this project. One of them is that the recommendation system only uses audio features and does not consider lyrics, listening history or individual preferences. Secondly, for EDA, only 7 out of 125 genres are considered, however user can add more genres in the streamlit app. Lastly, it is not verified with the actual user whether the recommendation is realistic or not. One further improvements in the future would include user listening history and preferences so that recommendations can be personalized. The recommendation system can also be evaluated through user feedback.
## AI Disclosure
ChatGPT was used as a supporting tool especially to convert my code in notebook to a streamlit code. I also used chatgpt for documentation of streamlit syntax. Moreover, I also used chatgpt to understand how cosine similarity work. Main prompts used are:
1) How does cosine similarity work in a content-based recommendation system?
2) How can I use this code so that it can be used by Streamlit application?
3) How can I change the color of default Streamlit to spotify green?
4) How can I make a card so that the song recommendation box looks pretty?

These are the main prompts. They are all mostly related to Streamlit.