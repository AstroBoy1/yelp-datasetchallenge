import pandas as pd
# https://stackoverflow.com/questions/30088006/loading-a-file-with-more-than-one-line-of-json-into-pythons-pandas


def main():
    df = pd.read_json("yelp_dataset/yelp_academic_dataset_business.json", lines=True)
    print(df.head)


if __name__ == "__main__":
    main()
