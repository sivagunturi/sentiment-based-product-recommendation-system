#Import the flask module
from flask import Flask,  render_template, jsonify
import os
import sys

import pandas as pd 
import numpy as np
from scipy.sparse import csr_matrix

from surprise import SVD,  SlopeOne
from surprise import KNNBaseline, KNNBasic, KNNWithMeans, KNNWithZScore

from surprise import Dataset
from surprise import accuracy
from surprise import Reader
import os
from surprise.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

class RecommendationSystem:
    def __init__(self, path):
        self.data_path = path
        self.load_data()

    def  load_data(self):
        print("<--------- 1. Read csv file ----------->")
        self.reviews_df = pd.read_csv(self.data_path)
        print("<--------- 2. Clean data ----------->")
        self.reviews_df.dropna(subset=['reviews_username'], inplace=True)
        self.reviews_df.drop(['reviews_date'], axis=1,inplace=True)
        self.reviews_df.columns.str.strip()
        print("<--------- 3. Split data ----------->")
        self.data = Dataset.load_from_df(self.reviews_df[['reviews_username', 'name', 'reviews_rating']], Reader(rating_scale=(1, 5)))
        self.trainset, self.testset = train_test_split(self.data, test_size=0.3,random_state=10)
        bsl_options = {'method': 'als', 'n_epochs': 5, 'reg_u': 12, 'reg_i': 5 }
        sim_options={'name': 'pearson_baseline', 'user_based': False}
        print("<--------- 4. Run algorithm ----------->")
        self.algo_KNNBaseline = KNNBaseline(k=5,sim_options = sim_options , bsl_options = bsl_options)
        self.predictions_KNNBaseline = self.algo_KNNBaseline.fit(self.trainset).test(self.testset)
        self.rmse_KNNBaseline = accuracy.rmse(self.predictions_KNNBaseline)
        print("<--------- 5. KNNBaseline Algorithm accuracy = ----------->" + str(self.rmse_KNNBaseline))

    def recommend_product(self, user_name):
        items_purchased = self.trainset.ur[self.trainset.to_inner_uid(user_name)]

        print("<--------- 5. Predicting Choosen User has purchased the following items --------->")
        for items in items_purchased[0]: 
            print(self.algo_KNNBaseline.trainset.to_raw_iid(items))

        #getting K Neareset Neighbors for first item purchased by the choosen user
        KNN_Product = self.algo_KNNBaseline.get_neighbors(items_purchased[0][0], 15)

        recommendation_list = []
        for product_iid in KNN_Product:
            if not product_iid in items_purchased[0]: #user already has purchased the item
                purchased_item = self.algo_KNNBaseline.trainset.to_raw_iid(product_iid)
                recommendation_list.append(purchased_item)
        print(recommendation_list)

        print("<--------- 6. Loading sentiment model --------->")
        loaded_model = pickle.load(open('sentiment_model.pkl', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        final_list = {}
        for item in recommendation_list:
            temp = self.reviews_df.loc[self.reviews_df['name'] == item]
            review_text = ""
            for index, row in temp.iterrows():
                review_text = row['reviews_text'] + " " + review_text
            features = vectorizer.transform([review_text])
            prediction = loaded_model.predict(features)[0]
            probability = loaded_model.predict_proba(features)
            print("item =>" + item + " sentiment = >" + str(prediction))    
            if(prediction):
                final_list[item] = probability[0][1]
        
        sort_orders = sorted(final_list.items(), key=lambda x: x[1], reverse=True)
        result = jsonify({'recommendations': sort_orders})
        print("<--------- 7. Returning the recommendations --------->" + str(result))
        sys.stdout.flush()
        return result      

rsys = RecommendationSystem("./sample30.csv")


@app.route('/',  methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/recommend/<username>', methods=['GET'])
def recommend(username):
    print("<--------- route recommender call --------->")
    return rsys.recommend_product(username)


#Calls the run method, runs the app on port 5000
# Create the main driver function
# port = int(os.environ.get("PORT", 5001)) # <-----
# app.run(host='0.0.0.0', port=port)       # <-----

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 8080))
#     app.run(host='0.0.0.0', port=port, debug=True)