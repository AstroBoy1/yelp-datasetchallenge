# Michael Omori
# Data Science
# Yelp dataset challenge 2018

import pandas as pd
import seaborn as sns
import numpy as np
from numpy import linalg as LA
from flair.embeddings import WordEmbeddings
from flair.data import Sentence


# ## TODO: analyze users, analyze everything specifically for Arizona, function modularity

# ## Read JSON Data

def similarity(t1, t2, glove_embedding):
    glove_embedding.embed(t1)
    glove_embedding.embed(t2)
    input1, input2 = t1[0].embedding, t2[0].embedding
    a = np.asarray(input1)
    b = np.asarray(input2)
    dot = np.dot(a.flatten(), b.flatten())
    a_mag = LA.norm(a)
    b_mag = LA.norm(b)
    sim = dot / a_mag / b_mag
    return sim


def analyze_checkin():
    pass


def analyze_business():
    pass


def analyze_reviews():
    pass


def analyze_users():
    pass


def main():
    glove_embedding = WordEmbeddings('glove')

    df = pd.read_json("yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    print(df.head)

    print(len(df))

    # ## Arizona businesses comprise about 1/3 of the dataset
    # ## Why?

    arizona_df = df[df['state'] == "AZ"]
    len(arizona_df)


    # ## What are all of the food related words?
    # ## Use Glove word embeddings as a proxy

    restaurant_score = []

    food = Sentence("food")
    restaurant_words = ["food", "cafe", "restaurants", "coffee", "drinks", "beer", "bar"]
    bar_words = ["beer", "bar", "bars", "brew"]
    bar_df = pd.DataFrame()
    restaurant_words_objects = []
    for word in restaurant_words:
        restaurant_words_objects.append(Sentence(word))

    bar_indices = []
    for i in range(0, len(df)):
        t = df['categories'][i]
        if t:
            t = t.lower()
            for word in bar_words:
                if word in t:
                    bar_indices.append(i)

    len(bar_indices)

    # ## 36k Bars and breweries

    len(bar_indices) / len(df)


    bars_df = df.iloc[bar_indices]

    # ## 20% of the businesses are bars

    for i in range(0, len(df)):
        best_score = 0
        score = 0
        if df['categories'][i]:
            for t in df['categories'][i].split(","):
                text = Sentence(t)
                glove_embedding.embed(text)
                for rwo in restaurant_words_objects:
                    score = similarity(rwo, text)
                    best_score = max(best_score, score)
        restaurant_score.append(best_score)
    df['restaurant'] = restaurant_score
    df.to_csv("restaurants.csv")

    df.sort_values(by=['restaurant'], ascending=False)

    df[df['restaurant'] > 0.9].to_csv("yelp_restaurants.csv")


    # ## Arizona restaurants

    yelp_restaurants_df = pd.read_csv("yelp_restaurants.csv")

    az_restaurants_df = yelp_restaurants_df[yelp_restaurants_df['state'] == 'AZ']
    az_restaurants_df.to_csv("AZ_restaurants.csv")

    len(az_restaurants_df)

    az_bars_df = bars_df[bars_df['state'] == 'AZ']
    az_bars_df.to_csv("AZ_bars.csv")

    len(az_bars_df)

    # ## Half the restaurants are bars in Arizona

    az_restaurants_df.head()


    # ## About 1/3 of Arizona businesses are restaurants in this dataset

    # ## What the embeddings physically look like

    # now check out the embedded tokens.
    # for token in sentence:
    #     print(token)
    #     print(token.embedding)


    # ## Where are the businesses located?

    food_columns = df[df['categories'].str.contains("Food")==True]
    print("Number of restaurants", len(food_columns))
    df['city'].value_counts()


    # ## Count of restaurants and count of total businesses

    restaurants_df = df[df['restaurant'] >= 1]

    len(restaurants_df)

    len(df)

    len(restaurants_df) / len(df)


    # ## 15 % of the businesses are restaurants
    # ## Most are located in Toronto, Las Vegas, and Phoenix

    restaurants_df['city'].value_counts()


    # ## Number of checkins during each day

    df_checkin = pd.read_json("yelp_dataset/yelp_academic_dataset_checkin.json", lines=True)
    print(df_checkin.head)

    df_checkin_AZ_restaurants = pd.merge(az_restaurants_df, df_checkin, on='business_id', how='inner')

    len(df_checkin_AZ_restaurants)

    df_checkin_AZ_restaurants.head()


    # ## Arizona restaurant checkin data
    # ## 14k

    day_counts = {'Mon': 0, 'Tue': 0, 'Wed': 0, 'Thu': 0, 'Fri': 0, 'Sat': 0, 'Sun': 0}
    time_counts = {x:0 for x in range(24)}

    day_counts_az = {'Mon': 0, 'Tue': 0, 'Wed': 0, 'Thu': 0, 'Fri': 0, 'Sat': 0, 'Sun': 0}
    time_counts_az = {x: 0 for x in range(24)}

    c = 0
    num_rows = len(df_checkin_AZ_restaurants)
    num_rows = 100
    for i in range(0, num_rows):
        time_checkins = df_checkin_AZ_restaurants['time'][i]
        # print(time_checkins)
        for k, v in time_checkins.items():
            print("\n", "Key:", k)
            print("Value:", v)
            day_count = k.split('-')
            print("Day count:", day_count)
            time_counts_az[int(day_count[1])] += v
            print(int(day_count[1]))
            day_counts_az[day_count[0]] += v

    c = 0
    num_rows = len(df_checkin)
    print("number of rows", num_rows)
    for i in range(0, len(df_checkin)):
        time_checkins = df_checkin['time'][i]
        for k, v in time_checkins.items():
            day_count = k.split('-')
            time_counts[int(day_count[1])] += v
            day_counts[day_count[0]] += v

    print("Day counts", day_counts)

    print("time counts", time_counts)


    # ## Checkins are higher on weekends
    # ## Lowest on Tuesdays

    day_counts_df = pd.DataFrame()
    day_counts_df['day'] = day_counts.keys()
    day_counts_df['counts'] = day_counts.values()
    sns.set(style="whitegrid")
    ax = sns.barplot(x="day", y="counts", data=day_counts_df)


    # ## Arizona restaurant checkins each day

    az_day_counts_df = pd.DataFrame()
    az_day_counts_df['day'] = day_counts_az.keys()
    az_day_counts_df['counts'] = day_counts_az.values()
    sns.set(style="whitegrid")
    ax = sns.barplot(x="day", y="counts", data=az_day_counts_df)


    # ## Number of checkins at restaruants through the day

    time_counts_df = pd.DataFrame()
    time_counts_df['time'] = time_counts.keys()
    time_counts_df['counts'] = time_counts.values()
    sns.set(style="whitegrid")
    ax = sns.barplot(x="time", y="counts", data=time_counts_df)


    # ## Arizona restaurant checkins through the day

    time_counts_df_az = pd.DataFrame()
    time_counts_df_az['time'] = time_counts_az.keys()
    time_counts_df_az['counts'] = time_counts_az.values()
    sns.set(style="whitegrid")
    ax = sns.barplot(x="time", y="counts", data=time_counts_df_az)

    print("Time counts", time_counts)

    cs = 100000
    count = 0
    fn = "yelp_gcs/yelp_academic_dataset_review.csv"
    stars = {x:0 for x in range(0, 6)}
    for chunk in pd.read_csv(fn, chunksize = cs):
        count += cs
        print(count)
        for index, row in chunk.sample(frac=0.1, replace=False, random_state=1).iterrows():
            try:
                stars[row['stars']] += 1
            except KeyError:
                print(row['stars'])


    # ## 6,000,000 rows of reviews

    print("Stars", stars)

    star_counts_df = pd.DataFrame()
    star_counts_df['stars'] = stars.keys()
    star_counts_df['counts'] = stars.values()
    sns.set(style="whitegrid")
    ax = sns.barplot(x="stars", y="counts", data=star_counts_df)


# ## 600k reviews

# ## More 5 star reviews, but also a lot of 1 star reviews

if __name__ == "__main__":
    main()
