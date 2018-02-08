from dotenv import find_dotenv, load_dotenv
import os

load_dotenv(find_dotenv())

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
YELP_API_KEY = os.environ['YELP_API_KEY']
