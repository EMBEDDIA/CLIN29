import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import iterparse, parse
from lxml import html
import pandas as pd

def xml_to_pandas(path, output):
    output_path = 'data/csv/' + output  + '.csv'
    if os.path.exists(output_path):
        os.remove(output_path)
    with open(path) as f:
        doc = html.fromstring(f.read())

    all_data = []
    for elem in doc:
        doc = [elem.attrib['id'], elem.attrib['gender'], elem.text]
        all_data.append(doc)
    df = pd.DataFrame(all_data, columns=['id', 'gender', 'text'])
    print(df.shape)

    df.to_csv(output_path, encoding="utf8", sep="\t", index=False)
    return df


def concat_corpora(corpora_list, output):
    all_corpora = []
    for corpus in corpora_list:
        corpus.reset_index(drop=True, inplace=True)
        all_corpora.append(corpus)
    df = pd.concat(all_corpora)
    print(df.shape)
    df.to_csv(output, encoding="utf8", sep="\t", index=False)
    return df


#df_news = xml_to_pandas('./data/test/GxG_News.txt', 'news_test')
#df_twitter = xml_to_pandas('./data/test/GxG_Twitter.txt', 'twitter_test')
#df_youtube = xml_to_pandas('./data/test/GxG_YouTube_test.txt', 'youtube_test')
df_surprise = xml_to_pandas('./data/test/GxG_KB.clean.txt', 'surprise_test_2')

#df_news = xml_to_pandas('./data/train/GxG_News.txt', 'news_train')
#df_twitter = xml_to_pandas('./data/train/GxG_Twitter.txt', 'twitter_train')
#df_youtube = xml_to_pandas('./data/train/GxG_YouTube.txt', 'youtube_train')
#concat_corpora([df_youtube, df_news], './data/csv/train_youtube_news.csv')

#df_news = xml_to_pandas('./data/test/GxG_News.txt', 'news_test')
#df_twitter = xml_to_pandas('./data/test/GxG_Twitter.txt', 'twitter_test')
#df_youtube = xml_to_pandas('./data/test/GxG_YouTube_test.txt', 'youtube_test')
#concat_corpora([df_news, df_twitter, df_youtube], './data/csv/test.csv')



