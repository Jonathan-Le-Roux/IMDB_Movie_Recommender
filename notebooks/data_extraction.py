import numpy as np
import pandas as pd


def download_all_data():
    import os
    import requests
    import gzip
    import shutil
    from bs4 import BeautifulSoup


    URL = "https://datasets.imdbws.com/" 

    FILENAMES = ['name.basics.tsv', 'title.akas.tsv', 'title.crew.tsv', \
                    'title.episode.tsv', 'title.principals.tsv', 'titl.ratings.tsv']
    
    response = requests.get(URL)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Delete old data
    if os.path.exists("../data"):
        shutil.rmtree("../data")

    # Create a fresh data directory
    os.makedirs("../data")

    # Extract the files
    for filename in FILENAMES:
        
        # Construct download URL
        file_url = URL + filename + ".gz"

        # Download compressed file
        response = requests.get(file_url, stream=True)
        response.raise_for_status()

        gz_path = f"../data/{filename}.gz"
        tsv_path = f"../data/{filename}"

        with open(gz_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024*1024*5):
                if chunk:
                    file.write(chunk)

        # Extract gzip file
        with gzip.open(gz_path, "rb") as compressed:
            with open(tsv_path, "wb") as extracted:
                shutil.copyfileobj(compressed, extracted)

        # Remove compressed file
        os.remove(gz_path)

        print(f"Downloaded and extracted: {filename}")

def preprocess_movie_basics():
    #read data in
    movie_basics = pd.read_csv("../data/title.basics.tsv" ,sep = '\t')
    
    #filter to only movies
    movie_basics = movie_basics[movie_basics['titleType']== 'movie'].reset_index(drop = True)

    #drop uninformative columns
    movie_basics = movie_basics.drop(columns = ['titleType', 'endYear', 'primaryTitle'])

    #reframe null_values to common convention
    movie_basics = movie_basics.replace({"\\N": np.nan})

    #convert to correct data types
    movie_basics['startYear']= movie_basics['startYear'].astype(float)
    movie_basics['runtimeMinutes']= movie_basics['runtimeMinutes'].astype(float)

    #we will need a title in order to recommend a movie:
    movie_basics = movie_basics.dropna(subset= ["originalTitle"])

    return movie_basics

def preprocess_title_ratings():
    ratings = pd.read_csv('../data/title.ratings.tsv', sep = '\t')
    return ratings

def preprocess_crew():
    crew = pd.read_csv('../data/title.crew.tsv', sep = '\t')
    crew =  crew.replace({'\\N': np.nan})
    return crew

def preprocess_names():
    crew_names = pd.read_csv("../data/name.basics.tsv", sep = '\t')
    crew_names = crew_names.dropna()
    return crew_names.loc[:,['nconst','primaryName']]

def preprocess_titles():
    titles_info = pd.read_csv('../data/title.akas.tsv', sep = '\t')
    titles_info = titles_info.replace({'\\N': np.nan})
    titles_info = titles_info.dropna(subset = ['region'])
    titles_info= titles_info.groupby('titleId').size()
    titles_info = pd.DataFrame(titles_info).reset_index()
    titles_info.columns = ['tconst', 'regionCount']
    return titles_info



def preprocess_data():
    
    #Preprocess indidividual files 
    movie_basics = preprocess_movie_basics()
    ratings = preprocess_title_ratings()
    crew = preprocess_crew()
    title_info = preprocess_titles()

    #Combine files into massive dataframe
    movie_with_ratings = movie_basics.merge(ratings, how = 'left')
    movie_with_ratings_and_crew = movie_with_ratings.merge(crew, how= 'left')
    complete_movie_df = movie_with_ratings_and_crew.merge(title_info, how = 'left')


    return complete_movie_df

    