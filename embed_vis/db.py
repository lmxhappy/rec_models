from pymongo import MongoClient


# mongo  10.30.54.33/admin -u zhaoqingxin -p 213518
class Mongo(object):
    def __init__(self,
                 user=None,
                 password=None,
                 host=None,
                 database=None,
                 collections=None):
        url = "mongodb://" + user + ":" + password + "@" + host
        self.client = MongoClient(url)
        self.collection = self.client[database][collections]

    def insert(self, data):
        return self.collection.insert(data)

    def find(self, query={}, keys=[]):
        if len(keys):
            return self.collection.find(query, keys)
        else:
            return self.collection.find(query)
    
    def find_one(self, query={}, keys=[]):
        if len(keys):
            return self.collection.find_one(query, keys)
        else:
            return self.collection.find_one(query)
    
    def update(self, query={}, update={}, upsert=False, manipulate=False, multi=False, check_keys=True):
        return self.collection.update(query, {"$set":update}, upsert=upsert, manipulate=manipulate, multi=multi, check_keys=check_keys)
    
    def remove(self, query={}):
        return self.collection.remove(query)