
from nltk import PorterStemmer
import collections
import pymongo
import math
import re


class search_module:

    def __init__(self):
        client = pymongo.MongoClient(host='localhost', port=27017)
        dbs = client.TTDS
        self.stemer = PorterStemmer()
        self.dict = collections.defaultdict(dict)
        self.data = dbs.recipe
        self.inverted_index = dbs.index
        self.token = []

    def tfidf_rank(self, doc_set):
        if not self.token:
            return []
        rank_list = []
        for i in doc_set:
            score = 0.0
            for word in self.token:
                if i in self.dict[word].keys():
                    score += (1 + math.log10(len(self.dict[word][i]))) * math.log10(
                        660000 / len(self.dict[word]))
            rank_list.append((i, score))
        rank_list.sort(key=lambda x: x[1], reverse=True)
        return rank_list[:10]

    def BM25_rank(self, query, doc_set):
        avg_word_num = 500.0
        para_k1 = 1
        para_b = 0.75
        rank_list = []
        for i in doc_set:
            score = 0.0
            for word in self.token:
                if i in self.dict[word].keys():
                    IDF = math.log10(1000000 / len(self.dict[word]))
                    TF = 1 + math.log10(len(self.dict[word][i]))
                    UP = (para_k1 + 1) * TF
                    DOWN = TF + para_k1 * \
                        ((1 - para_b) + para_b * (1000000 / avg_word_num))
                    score += UP / DOWN * IDF
            rank_list.append((i, score))
        rank_list.sort(key=lambda x: x[1], reverse=True)
        return rank_list[:10]

    def phrase_query(self, str):
        l = str.split()
        doclist = []
        a = self.stemer.stem(l[0].strip(), to_lowercase=True)
        b = self.stemer.stem(l[1].strip(), to_lowercase=True)
        res1 = self.inverted_index.find_one({'term': a})
        res2 = self.inverted_index.find_one({'term': b})
        if not res1 or not res2:
            return []
        self.token.append(a)
        self.token.append(b)
        if a not in self.dict.keys():
            self.dict.update({a: res1['docList']})
        if b not in self.dict.keys():
            self.dict.update({b: res2['docList']})

        first_list = res1['docList'].keys()
        second_list = res2['docList'].keys()
        common_list = [i for i in first_list if i in second_list]
        for i in common_list:
            x = [i for x in self.dict[a][i] if x + 1 in self.dict[b][i]]
            if x:
                doclist.append(x[0])
        return doclist

    def getDocSet(self, query):
        flag = 0
        self.token.clear()
        doc_set = set()
        if "or" in query:
            flag = 1
        else:
            flag = 0
        query = query.replace("or", "")
        query = query.replace("and", "")
        phrase = re.search(r"\"(.+)\"", query)
        query = re.sub(r"\"(.+)\"", "", query)
        if phrase:
            doc_set = set(self.phrase_query(phrase.group(1)))

        terms = query.split(" ")
        for term in terms:
            term = self.stemer.stem(term)
            res = self.inverted_index.find_one({'term': term})
            if not res:
                continue
            word = res['term']
            self.token.append(word)
            if word not in self.dict.keys():
                self.dict.update({word: res['docList']})

            if flag == 1:
                doc_set = doc_set.union(set(self.dict[term].keys()))
            else:
                if not doc_set:
                    doc_set = set(self.dict[term].keys())
                else:
                    doc_set = doc_set.intersection(set(self.dict[term].keys()))

        # print(terms)
        # print(phrase)
        # print(self.token)

        return doc_set

    def getContent(self, docId):
        return self.data.find_one({'id': int(docId)}, {'_id': 0})

    def convert_db_to_json(self, db_result):
        json_data = {}
        json_data['article_id'] = db_result["id"]
        json_data['title'] = db_result['title']
        json_data['canonical_link'] = "https://" + \
            db_result['link'].replace("www.", "")
        json_data['url'] = "https://" + db_result['link'].replace("www.", "")
        json_data['summary'] = db_result["NER"]
        json_data['bert'] = db_result["bert"]
        return json_data

    def search(self, query):
        docset = self.getDocSet(query.lower())
        docList = self.tfidf_rank(docset)
        # print(docList[:5])
        results = []
        for docid in docList:
            data = self.getContent(docid[0])
            json_data = self.convert_db_to_json(data)
            results.append(json_data)
        return results
