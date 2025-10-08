import os
from typing import Any, Dict, Optional

import tweepy


class TwitterClient:
    """Simple Twitter client to retrieve user's last post."""

    def __init__(self):
        """Initialize Twitter client with credentials from environment variables."""
        self.api_key = os.getenv("TWITTER_API_KEY")
        self.api_secret = os.getenv("TWITTER_API_SECRET")
        self.access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        self.access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.username = os.getenv("TWITTER_USERNAME")

        # Validate required credentials
        required_creds = [
            self.api_key,
            self.api_secret,
            self.access_token,
            self.access_token_secret,
            self.bearer_token,
        ]

        if not all(required_creds):
            raise ValueError(
                "Missing required Twitter API credentials. Please check your .env file.\n"
                "Required: TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN, "
                "TWITTER_ACCESS_TOKEN_SECRET, TWITTER_BEARER_TOKEN"
            )

        # Initialize Tweepy client
        self.client = tweepy.Client(
            bearer_token=self.bearer_token,
            consumer_key=self.api_key,
            consumer_secret=self.api_secret,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
            wait_on_rate_limit=True,
        )

    def get_user_id(self, username: Optional[str] = None) -> str:
        """Get user ID from username."""
        target_username = username or self.username
        if not target_username:
            raise ValueError(
                "No username provided. Set TWITTER_USERNAME in .env or pass username parameter."
            )

        # Remove @ if present
        target_username = target_username.lstrip("@")

        try:
            user = self.client.get_user(username=target_username)
            return user.data.id
        except Exception as e:
            raise Exception(f"Failed to get user ID for @{target_username}: {str(e)}")

    def get_last_post(self, username: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve the last post from the specified user.

        Args:
            username: Twitter username (without @). If None, uses TWITTER_USERNAME from .env

        Returns:
            Dictionary containing post information or None if no posts found
        """
        try:
            # Get user ID
            user_id = self.get_user_id(username)

            # Get user's tweets (excluding retweets and replies)
            tweets = self.client.get_users_tweets(
                id=user_id,
                max_results=5,  # Twitter API requires min 5, max 100
                exclude=["retweets", "replies"],
                tweet_fields=["created_at", "public_metrics", "context_annotations"],
            )

            if not tweets.data:
                return None

            tweet = tweets.data[0]

            return {
                "id": tweet.id,
                "text": tweet.text,
                "created_at": tweet.created_at,
                "public_metrics": tweet.public_metrics,
                "url": f"https://twitter.com/{username or self.username}/status/{tweet.id}",
            }

        except Exception as e:
            raise Exception(f"Failed to retrieve last post: {str(e)}")

    def get_last_post_formatted(self, username: Optional[str] = None) -> str:
        """
        Get the last post in a formatted string.

        Args:
            username: Twitter username (without @). If None, uses TWITTER_USERNAME from .env

        Returns:
            Formatted string with post information
        """
        post = self.get_last_post(username)

        if not post:
            return "No posts found for this user."

        formatted = f"""
🐦 Last Twitter Post from @{username or self.username}:

📝 Text: {post['text']}
📅 Posted: {post['created_at']}
🔗 URL: {post['url']}

📊 Metrics:
  • Likes: {post['public_metrics']['like_count']}
  • Retweets: {post['public_metrics']['retweet_count']}
  • Replies: {post['public_metrics']['reply_count']}
  • Quotes: {post['public_metrics']['quote_count']}
"""
        return formatted.strip()
