# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymongo

class ActionDefaultFallback(Action):
    def name(self) -> Text:
        return "action_default_fallback"

    def __init__(self):
        # Connect to the MongoDB database
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["solyield_db"]
        self.collection = self.db["docs"]
        self.knowledge_base = [doc["text"] for doc in self.collection.find()]
        self.vectorizer = TfidfVectorizer().fit_transform(self.knowledge_base)
        self.threshold = 0.5
    
    def find_most_similar_chunk(self, query):
        query_vec = TfidfVectorizer().fit(self.knowledge_base).transform([query])
        similarities = cosine_similarity(query_vec, self.vectorizer).flatten()
        most_similar_index = similarities.argmax()
        if similarities[most_similar_index] >= self.threshold:
            return self.knowledge_base[most_similar_index]
        else:
            return "I'm sorry, I couldn't find a relevant answer with questions relevant to solar panels. Can you please clarify your question?"
    
    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        question = tracker.latest_message['text']
        print("Question: ", question)
        answer = self.find_most_similar_chunk(question)
        dispatcher.utter_message(text=answer)

        return []

class ActionConsumptionDetails(Action):

    def name(self) -> Text:
        return "action_consumption_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        # Connect to the MongoDB database
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client["solyield_db"]
        collection = db["consumer"]

        username= tracker.get_slot("user_id")
        print(username)

        # Query the database for user data based on user ID
        user_data = collection.find_one({"user_id": username})
        print(user_data)

        if user_data:
            # If user data is found, extract relevant information
            name = user_data.get("name")
            consumption = user_data.get("consumption")
            bill = user_data.get("bill")

            # Respond with the fetched data
            dispatcher.utter_message(f"Hello {name}! Your consumption details are: {consumption}")
        else:
            # If user data is not found, respond with a default message
            dispatcher.utter_message("Sorry, I couldn't find your data in the database.")

        # Close the database connection
        client.close()

        return []
