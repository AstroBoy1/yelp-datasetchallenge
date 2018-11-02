import json
fn = 'yelp_dataset/yelp_academic_dataset_business.json'
with open(fn, encoding='utf8') as f:
    data = json.load(f)
