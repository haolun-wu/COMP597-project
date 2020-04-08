import gzip
import json
import argparse
import pandas as pd

from utils import *


def generate_brand_dict(file_name):
    """
    This code shows the process used to categorize the brands fairness segments. Note that it was run on the raw data
    directly and won't work if you try to simply run it now
    :param file_name:
    :return: dictionary of { "ProductID": Fairness Score (1-10) }
    """
    def load_data(file_name):

        count_i = 0
        data = []
        with gzip.open(file_name) as fin:
            for l in fin:
                d = json.loads(l)
                count_i += 1
                data.append(d)

            # break if reaches the 100th line
            # if (head is not None) and (count > head):
            # break
        return data

    def process_data(file_name):

        data = load_data(file_name + '.json.gz')
        print(len(data))

        df = pd.DataFrame.from_dict(data)

        print(len(df))
        print(df)

        df_new = df
        df_new.rename(
            columns={'reviewerID': 'userID', 'asin': 'itemID', 'overall': 'rating', 'reviewTime': 'timestamp'},
            inplace=True)
        order = ['userID', 'itemID', 'rating', 'timestamp']
        df_new = df_new[order]

        print(df_new[:5])
        return df_new

    def process_data_meta(file_name):

        data = load_data(file_name + '.json.gz')
        print(len(data))

        df = pd.DataFrame.from_dict(data)

        print(len(df))

        df_new = df
        df_new.rename(columns={'asin': 'itemID'}, inplace=True)
        order = ['itemID', 'brand']
        df_new = df_new[order]

        print(df_new[:5])
        return df_new

    df2 = process_data_meta(file_name)

    dic = df2.set_index('itemID')['brand'].to_dict()
    count = {}
    for ch in dic.values():
        count[ch] = 1 + count.get(ch, 0)

    tmp = sorted(count.items(), key=lambda d: d[1], reverse=True)
    scale = tmp
    # print(tmp)
    ls = len(tmp)
    print("How many different brands in total?", ls)

    s, r = divmod(ls, 10)

    for i in range(ls):

        scale[i] = list(scale[i])

        if s * 0 <= i < s * 1:
            scale[i][1] = 10
        elif s * 1 <= i < s * 2:
            scale[i][1] = 9
        elif s * 2 <= i < s * 3:
            scale[i][1] = 8
        elif s * 3 <= i < s * 4:
            scale[i][1] = 7
        elif s * 4 <= i < s * 5:
            scale[i][1] = 6
        elif s * 5 <= i < s * 6:
            scale[i][1] = 5
        elif s * 6 <= i < s * 7:
            scale[i][1] = 4
        elif s * 7 <= i < s * 8:
            scale[i][1] = 3
        elif s * 8 <= i < s * 9:
            scale[i][1] = 2
        else:
            scale[i][1] = 1

    new_dic = dict(scale)

    for i in range(len(df2)):
        df2['brand'][i] = new_dic[df2['brand'][i]]
    print("finished 1")

    dic2 = df2.set_index('itemID')['brand'].to_dict()
    print("finished 2")

    return dic2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=BEAUTY, help='One of {cloth, beauty, cell, cd}')
    parser.add_argument('--name', type=str, default='train_agent', help='directory name.')
    args = parser.parse_args()
    args.log_dir = TMP_DIR[args.dataset] + '/' + args.name
    generate_brand_dict(args.log_dir)


if __name__ == '__main__':
    main()
