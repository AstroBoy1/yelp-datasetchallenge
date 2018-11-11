import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# https://seaborn.pydata.org/tutorial.html
# Folium for map visualization

# https://stackoverflow.com/questions/30088006/loading-a-file-with-more-than-one-line-of-json-into-pythons-pandas
# https://www.dataquest.io/blog/python-json-tutorial/
# Dataframe subsetting: https://medium.com/dunder-data/selecting-subsets-of-data-in-pandas-6fcd0170be9c
# Dataframe string: https://stackoverflow.com/questions/11350770/pandas-dataframe-select-by-partial-string


def main():
    df = pd.read_json("yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    print(df.head)
    food_columns = df[df['categories'].str.contains("Food")==True]
    print("Number of restaurants", len(food_columns))
    df['city'].value_counts()


if __name__ == "__main__":
    main()
