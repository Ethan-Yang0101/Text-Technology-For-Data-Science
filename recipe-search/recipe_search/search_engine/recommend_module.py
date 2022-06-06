
import numpy as np


class recommend_module:
    
    def __init__(self, search_module):
        self.search_module = search_module
        self.user_history = []
        self.user_embedding = np.zeros(768)
        self.user_tokens = []
        
    def add_user_history(self, result):
        result['bert'] = self.generate_embedding(result)
        if len(self.user_history) < 10:
            self.user_history.append(result)
            self.update_user_embedding()
            self.update_user_tokens()
        else:
            self.user_history.pop(0)
            self.user_history.append(result)
            self.update_user_embedding()
            self.update_user_tokens()
        return
    
    def generate_embedding(self, result):
        embedding = np.array([float(num) for num in result['bert'][1:-1].split(', ')])
        return embedding
        
    def update_user_embedding(self):
        user_embedding = np.zeros(768)
        for idx, result in enumerate(self.user_history[::-1]):
            user_embedding += result['bert'] * (0.9 ** idx)
        self.user_embedding = user_embedding
        return
    
    def update_user_tokens(self):
        user_tokens = []
        for result in self.user_history:
            tokens = result['title'].split(' ')
            user_tokens.extend(tokens)
        self.user_tokens = list(set(user_tokens))
        return
    
    def compute_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def recommend(self):
        unique_docset, rough_results = [], [] 
        for token in self.user_tokens:
            docset = self.search_module.getDocSet(token.lower())
            doclist = self.search_module.tfidf_rank(docset)
            docset = [doc[0] for doc in doclist]
            unique_docset.extend(docset)
        unique_docset = set(unique_docset)
        for docid in unique_docset:
            data = self.search_module.getContent(docid)
            embedding = self.generate_embedding(data)
            score = self.compute_similarity(embedding, self.user_embedding)
            rough_results.append((score, data))
        results = sorted(rough_results, key=lambda x: x[0], reverse=True)[:10]
        results = [self.search_module.convert_db_to_json(result[1]) for result in results]
        return results
    