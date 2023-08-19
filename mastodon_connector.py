from mastodon import Mastodon

import re

import configparser

# Create a configparser instance and read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Retrieve Mastodon API credentials from the config file
client_id = config['mastodon']['client_id']
client_secret = config['mastodon']['client_secret']
access_token = config['mastodon']['access_token']
api_base_url = config['mastodon']['api_base_url']

# Create a Mastodon client instance using the retrieved credentials
mastodon = Mastodon(
    client_id=client_id,
    client_secret=client_secret,
    access_token=access_token,
    api_base_url=api_base_url
)

# Define a function to search for posts with a specific hashtag
def search_tag(tag):
    mylist = []

    # Get the timeline for a specific hashtag
    hashtag_timeline = mastodon.timeline_hashtag(tag)
    
    # Define a regular expression pattern to remove HTML tags from post content
    pattern = re.compile('<.*?>')

    # Iterate through the posts in the hashtag timeline
    for post in hashtag_timeline:
        # Remove HTML tags from the post content
        result = re.sub(pattern, '', post['content'])
        # Append the cleaned content to the list
        mylist.append(result)

    # Return the list of cleaned post content
    return mylist

# Call the search_tag function to retrieve posts with the "bitcoin" hashtag
listposts = search_tag("bitcoin")

# Print each cleaned post content
for post in listposts:
    print(f"{post}")
