import sys

from dotenv import load_dotenv
from twitter_client import TwitterClient

load_dotenv()


def main():
    """Main function to retrieve and display the last Twitter post."""
    try:
        # Initialize Twitter client
        twitter = TwitterClient()

        # Get username from command line argument or use default from .env
        username = sys.argv[1] if len(sys.argv) > 1 else None

        print("🔍 Retrieving your last Twitter post...")
        print("-" * 50)

        # Get and display the last post
        formatted_post = twitter.get_last_post_formatted(username)
        print(formatted_post)

    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("\n📝 Please make sure your .env file contains the following variables:")
        print("   TWITTER_API_KEY=your_api_key")
        print("   TWITTER_API_SECRET=your_api_secret")
        print("   TWITTER_ACCESS_TOKEN=your_access_token")
        print("   TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret")
        print("   TWITTER_BEARER_TOKEN=your_bearer_token")
        print("   TWITTER_USERNAME=your_username (optional)")

    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
