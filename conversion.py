# -*- coding: utf-8 -*-
"""Convert the Yelp Dataset Challenge dataset from json format to csv.

For more information on the Yelp Dataset Challenge please visit http://yelp.com/dataset_challenge

"""
import argparse
import collections
import csv
import simplejson as json
import sys, getopt


def read_and_write_file(json_file_path, csv_file_path, column_names):
    """Read in the json dataset file and write it out to a csv file, given the column names."""
    # count = 0
    with open(csv_file_path, 'w', newline='') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path, encoding='utf-8') as fin:
            for index, line in enumerate(fin):
                # print(index)
                line_contents = json.loads(line)
                # print("Json contents:", line_contents)
                # print("CSV contents:", get_row(line_contents, column_names))
                # count += 1
                # if count > 2:
                #     break
                s = get_row(line_contents, column_names)
                s = [val.encode('unicode_escape').decode() if isinstance(val, str) else val for val in s]
                try:
                    csv_file.writerow(s)
                except UnicodeEncodeError:
                    print(s)
                    print("Unicode Error")


def get_superset_of_column_names_from_file(json_file_path):
    """Read in the json dataset file and return the superset of column names."""
    column_names = set()
    with open(json_file_path, encoding='utf-8') as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                set(get_column_names(line_contents).keys())
            )
    return column_names


def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.

    Example:

        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }

        will return: ['a.b', 'a.c']

    These will be the column names for the eventual csv file.

    """
    column_names = []
    # for k, v in line_contents.iteritems():
    for k, v in line_contents.items():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                get_column_names(v, column_name).items()
            )
        else:
            column_names.append((column_name, v))
    return dict(column_names)


def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.

    Example:

        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'

        will return: 2

    """
    if not d:
        return None
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)


def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
            line_contents,
            column_name,
        )
        # print("Column name", column_name)
        # print("Line value", line_value)
        row.append(line_value)
        # if isinstance(line_value, str):
        #     row.append('{0}'.format(line_value.encode('utf-8')))
        # elif line_value is not None:
        #     row.append('{0}'.format(line_value))
        # else:
        #     row.append('')
    return row


def main(argv):
    # json_file = "yelp_dataset/yelp_academic_dataset_business.json"
    json_file = "yelp_dataset/yelp_academic_dataset_photo.json"
    try:
        opts, args = getopt.getopt(argv, "hi:", ["ifile="])
    except getopt.GetoptError:
        print('conversion.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            json_file = arg

    csv_file = '{0}.csv'.format(json_file.split('.json')[0])

    column_names = get_superset_of_column_names_from_file(json_file)
    read_and_write_file(json_file, csv_file, column_names)
    print("Finished", json_file)


if __name__ == '__main__':
    """Convert a yelp dataset file from json to csv."""
    """TODO: Convert all the json files"""

    # parser = argparse.ArgumentParser(
    #         description='Convert Yelp Dataset Challenge data from JSON format to CSV.',
    #         )
    #
    # parser.add_argument(
    #         'json_file',
    #         type=str,
    #         help='The json file to convert.',
    #         )
    #
    # args = parser.parse_args()
    #
    # json_file = args.json_file
    main(sys.argv[1:])
