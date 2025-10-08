# Twitter Last Post Retrieval

A simple Python solution to retrieve your last Twitter post using the Twitter API v2.

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with your Twitter API credentials:

```env
# Twitter API v2 Credentials
# Get these from https://developer.twitter.com/en/portal/dashboard
TWITTER_API_KEY=your_api_key_here
TWITTER_API_SECRET=your_api_secret_here
TWITTER_ACCESS_TOKEN=your_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here
TWITTER_BEARER_TOKEN=your_bearer_token_here

# Optional: Your Twitter username (without @)
TWITTER_USERNAME=your_username_here
```

### 3. Get Twitter API Credentials

1. Go to [Twitter Developer Portal](https://developer.twitter.com/en/portal/dashboard)
2. Create a new app or use an existing one
3. Generate API keys and access tokens
4. Make sure your app has read permissions for user tweets

## Usage

### Basic Usage (using username from .env)

```bash
python main.py
```

### Specify a Different Username

```bash
python main.py username
```

### Example Output

```
🔍 Retrieving your last Twitter post...
--------------------------------------------------
🐦 Last Twitter Post from @your_username:

📝 Text: This is my latest tweet!
📅 Posted: 2024-01-15 10:30:00+00:00
🔗 URL: https://twitter.com/your_username/status/1234567890

📊 Metrics:
  • Likes: 42
  • Retweets: 5
  • Replies: 3
  • Quotes: 1
```

## Features

- ✅ Retrieves the most recent original tweet (excludes retweets and replies)
- ✅ Shows engagement metrics (likes, retweets, replies, quotes)
- ✅ Provides direct link to the tweet
- ✅ Handles errors gracefully with helpful messages
- ✅ Supports command-line username specification
- ✅ Uses Twitter API v2 with proper authentication

## Error Handling

The script will show helpful error messages if:
- Missing API credentials
- Invalid username
- API rate limits
- Network issues
- No posts found

## Requirements

- Python 3.13+
- Twitter API v2 access
- Valid Twitter API credentials
