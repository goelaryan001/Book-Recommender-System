import os
import sys
import pickle
import streamlit as st
import numpy as np
from books_recommender.logger.log import logging
from books_recommender.config.configuration import AppConfiguration
from books_recommender.pipeline.training_pipeline import TrainingPipeline
from books_recommender.exception.exception_handler import AppException


@st.cache_resource
def load_pickle(path):
    return pickle.load(open(path, 'rb'))


class Recommendation:
    def __init__(self,app_config = AppConfiguration()):
        try:
            self.recommendation_config= app_config.get_recommendation_config()
        except Exception as e:
            raise AppException(e, sys) from e
        
    

    def fetch_poster(self,suggestion):
        try:
            book_name = []
            ids_index = []
            poster_url = []
            book_pivot = load_pickle(self.recommendation_config.book_pivot_serialized_objects)
            final_rating = load_pickle(self.recommendation_config.final_rating_serialized_objects)

            for book_id in suggestion:
                book_name.append(book_pivot.index[book_id])

            for name in book_name[0]: 
                ids = np.where(final_rating['title'] == name)[0][0]
                ids_index.append(ids)

            for idx in ids_index:
                url = final_rating.iloc[idx]['image_url']
                poster_url.append(url)

            return poster_url
        
        except Exception as e:
            raise AppException(e, sys) from e
        

    
    def recommend_book(self,book_name):
        try:
            books_list = []
            model = load_pickle(self.recommendation_config.trained_model_path)
            book_pivot = load_pickle(self.recommendation_config.book_pivot_serialized_objects)
            book_id = np.where(book_pivot.index == book_name)[0][0]
            distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=7 )

            # The queried book is its own nearest neighbor (distance 0); drop it before returning results
            neighbors = [(idx, dist) for idx, dist in zip(suggestion[0], distance[0]) if idx != book_id][:6]
            suggestion = np.array([[idx for idx, _ in neighbors]])
            similarity = [1 - dist for _, dist in neighbors]

            poster_url = self.fetch_poster(suggestion)

            for i in range(len(suggestion)):
                    books = book_pivot.index[suggestion[i]]
                    for j in books:
                        books_list.append(j)
            return books_list, poster_url, similarity
        
        except Exception as e:
            raise AppException(e, sys) from e
        

    
    def recommend_by_author(self, book_name, max_results=5):
        try:
            final_rating = load_pickle(self.recommendation_config.final_rating_serialized_objects)
            match = final_rating[final_rating['title'] == book_name]
            if match.empty:
                return [], []

            author = match.iloc[0]['author']
            same_author = final_rating[
                (final_rating['author'] == author) & (final_rating['title'] != book_name)
            ].drop_duplicates('title')

            books = same_author['title'].head(max_results).tolist()
            poster_url = same_author['image_url'].head(max_results).tolist()
            return books, poster_url

        except Exception as e:
            raise AppException(e, sys) from e



    def train_engine(self):
        try:
            obj = TrainingPipeline()
            obj.start_training_pipeline()
            st.text("Training Completed!")
            logging.info(f"Recommended successfully!")
        except Exception as e:
            raise AppException(e, sys) from e
        
    

    def recommendations_engine(self,selected_books):
        try:
            recommended_books, poster_url, similarity = self.recommend_book(selected_books)
            columns = st.columns(5)
            for col, book, url, sim in zip(columns, recommended_books, poster_url, similarity):
                with col:
                    st.text(book)
                    st.image(url)
                    st.caption(f"{sim:.0%} similar")
        except Exception as e:
            raise AppException(e, sys) from e



    def author_recommendations_engine(self, selected_books):
        try:
            author_books, author_posters = self.recommend_by_author(selected_books)
            if not author_books:
                return

            st.subheader("More by the same author")
            columns = st.columns(len(author_books))
            for col, book, url in zip(columns, author_books, author_posters):
                with col:
                    st.text(book)
                    st.image(url)
        except Exception as e:
            raise AppException(e, sys) from e



if __name__ == "__main__":
    st.header('End to End Books Recommender System')
    st.text("This is a collaborative filtering based recommendation system!")

    obj = Recommendation()

    #Training
    if st.button('Train Recommender System'):
        obj.train_engine()

    book_names = load_pickle(os.path.join('templates','book_names.pkl'))
    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names)
    
    #recommendation
    if st.button('Show Recommendation'):
        obj.recommendations_engine(selected_books)
        obj.author_recommendations_engine(selected_books)
