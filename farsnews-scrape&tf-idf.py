import requests
from bs4 import BeautifulSoup
from newspaper import Article
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import json, codecs
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


#این  بخش که کامنت شده، به همراه چند خط کد که در پایین تر قرار دارد،شرط شکستن حلقه هستند.
# def repetitive(links, urls):
#     for link in links:
#         if link['href'] in urls:
#             return True
    # return False


#بر اساس دسته بندی اخبار، آن ها را خراش می دهیم
def scrap_cat(cate: str):
    page = 0
    scraped_data = []
    url_list = []
    n = 0
    
    #در این شرط، به جای استفاده از دستور شکست حلقه، چون محتوای سایت زیاد میشود،سه تا از صفحات را بررسی می کنیم.
    #قرار دهیم True درصورت تمایل به استفاده از شکست حلقه باید شرط را

    while n<=2:
        page += 1
        n += 1

        main_page_url = f"https://www.farsnews.ir/{cate}?p={page}"

        html = requests.get(main_page_url).text

        soup = BeautifulSoup(html, "lxml")

        #قرار گرفتند a اخبار، داخل تگ
        #که کلاس آنها، همین کلاسی است که در کد زیر اشاره شده
        links = soup.find_all("a", {"class": "d-flex flex-column h-100 justify-content-between"})


        # استفاده از تابع شکست حلقه
        # if repetitive(links, url_list):
        #     break

        #حلقه برای دانلود و پارس اخبار این صفحه
        for link in tqdm(links):
            page_url = 'https://www.farsnews.ir' + link['href']

            url_list.append(link['href'])

            try:
                #در این 4 خط، اخبار را دانلود می کنیم و پارس میکنیم.
                article = Article(page_url)
                article.download()
                article.parse()
                #url و text و title یک دیکشنری ایجاد می کنیم با کلید های
                # و داخل آن ها، جزئیات مربوط را قرار می دهیم و به لیست، اضافه می کنیم
                scraped_data.append({'url': page_url, 'text': article.text, 'title': article.title})
            except:
                print(f"Failed to process page: {page_url}")


    #اگر بخواهیم داخل فایل، آنرا ذخیره کنیم، از کد زیر استفاده می کنیم.
    # corpus = pd.DataFrame(scraped_data)
    # corpus.to_csv(f'farsnews-{cate}.csv') 


    # انتخاب کردن عنوان ها و قرار دادن در این لیست، برای سرچ روی آن
    docs = [d['title'] for d in scraped_data]

    # tf-idf
    vectorizer = TfidfVectorizer()
    tfidf_docs = vectorizer.fit_transform(docs)
    # print(tfidf_docs.shape, len(vectorizer.vocabulary_))

    #ده تا اطلاعات مربوط را نشان می دهد
    list(vectorizer.vocabulary_.keys())[:10]

    # query
    query = 'برای ورزش'

    tfidf_query = vectorizer.transform([query])[0]
    # similarities
    cosines = []
    for d in tqdm(tfidf_docs):
        cosines.append(float(cosine_similarity(d, tfidf_query)))

    # sorting
    k = 10
    sorted_ids = np.argsort(cosines)
    for i in range(k):
        cur_id = sorted_ids[-i-1]
        print(docs[cur_id], cosines[cur_id])









if __name__ == '__main__':
    scrap_cat("scientific-academic")









