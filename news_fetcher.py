"""
News Fetcher Module for Gold Silver AI
Fetches latest news articles related to gold, silver, and macroeconomic factors
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    from gnews import GNews
except ImportError:
    GNews = None


class NewsFetcher:
    """Handles fetching news articles from Google News"""

    def __init__(self):
        if GNews is None:
            raise ImportError("GNews package is required. Install with: pip install gnews")

        try:
            # Initialize GNews client
            self.gnews = GNews(language='en', country='IN', period='1d', max_results=20)
        except Exception as e:
            raise ImportError(f"Failed to initialize GNews client: {e}")

        # Default query for gold/silver and macroeconomic factors
        self.default_query = "gold OR silver OR inflation OR rbi OR usd OR oil OR interest rate OR crude oil"

    def fetch_news(self, query: str = None, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch news articles based on query

        Args:
            query: Search query (optional, uses default if None)
            max_results: Maximum number of articles to fetch

        Returns:
            List of article dictionaries with standardized format
        """
        if query is None:
            query = self.default_query

        try:
            # Get news from GNews
            articles = self.gnews.get_news(query)

            # Standardize the format
            standardized_articles = []
            for article in articles[:max_results]:
                standardized_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'published_date': article.get('published date', ''),
                    'source': article.get('publisher', {}).get('title', ''),
                    'url': article.get('url', '')
                }
                standardized_articles.append(standardized_article)

            return standardized_articles

        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def fetch_latest_news(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch latest news articles using default query

        Args:
            max_results: Maximum number of articles to fetch

        Returns:
            List of latest articles
        """
        return self.fetch_news(max_results=max_results)


def fetch_news(query: str = None, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Convenience function to fetch news articles

    Args:
        query: Search query (optional)
        max_results: Maximum number of articles to fetch

    Returns:
        List of article dictionaries
    """
    fetcher = NewsFetcher()
    return fetcher.fetch_news(query, max_results)


if __name__ == "__main__":
    # Test the news fetcher
    print("Testing News Fetcher...")

    try:
        articles = fetch_news(max_results=5)

        print(f"Fetched {len(articles)} articles:")
        for i, article in enumerate(articles, 1):
            print(f"\n{i}. {article['title']}")
            print(f"   Source: {article['source']}")
            print(f"   Date: {article['published_date']}")
            print(f"   URL: {article['url'][:100]}...")

    except Exception as e:
        print(f"Error: {e}")