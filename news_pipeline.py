"""
News Pipeline Module for Gold Silver AI
Integrates news fetching, NLP processing, and database storage
"""

import sqlite3
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from news_fetcher import NewsFetcher
from news_nlp import NewsNLP


class NewsPipeline:
    """Handles the complete news intelligence pipeline"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            # Use absolute path based on this script's location
            project_root = Path(__file__).resolve().parent
            db_path = project_root / "gold_silver_data.db"

        self.db_path = Path(db_path)

        try:
            self.fetcher = NewsFetcher()
        except Exception as e:
            raise ImportError(f"Failed to initialize NewsFetcher: {e}")

        try:
            self.nlp = NewsNLP()
        except Exception as e:
            raise ImportError(f"Failed to initialize NewsNLP: {e}")

        try:
            self._init_db()
        except Exception as e:
            raise ImportError(f"Failed to initialize database: {e}")

    def _init_db(self):
        """Initialize the news_data table in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT,
                    factors TEXT,  -- JSON array of factors
                    sentiment REAL,
                    url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, title, source)
                )
            """)

            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_date ON news_data(date)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_factors ON news_data(factors)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_data(sentiment)")

            conn.commit()

    def fetch_and_process_news(self, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch news and process with NLP

        Args:
            max_results: Maximum number of articles to fetch

        Returns:
            List of processed articles
        """
        try:
            # Fetch raw news
            raw_articles = self.fetcher.fetch_latest_news(max_results=max_results)
            print(f"Fetched {len(raw_articles)} raw articles")

            # Process with NLP
            processed_articles = self.nlp.process_articles(raw_articles)
            print(f"Processed {len(processed_articles)} articles with NLP")

            # Filter relevant articles
            relevant_articles = self.nlp.filter_relevant_articles(processed_articles)
            print(f"Filtered to {len(relevant_articles)} relevant articles")

            return relevant_articles

        except Exception as e:
            print(f"Error in fetch_and_process_news: {e}")
            return []

    def save_to_database(self, articles: List[Dict[str, Any]]) -> int:
        """
        Save processed articles to database

        Args:
            articles: List of processed articles

        Returns:
            Number of articles saved
        """
        saved_count = 0

        with sqlite3.connect(self.db_path) as conn:
            for article in articles:
                try:
                    # Convert factors list to JSON string
                    factors_json = json.dumps(article.get('factors', []))

                    # Prepare data
                    article_data = {
                        'date': article.get('date', datetime.now().strftime('%Y-%m-%d')),
                        'title': article.get('title', ''),
                        'source': article.get('source', ''),
                        'factors': factors_json,
                        'sentiment': article.get('sentiment', 0.0),
                        'url': article.get('url', '')
                    }

                    # Insert or replace (handles duplicates)
                    conn.execute("""
                        INSERT OR REPLACE INTO news_data
                        (date, title, source, factors, sentiment, url)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        article_data['date'],
                        article_data['title'],
                        article_data['source'],
                        article_data['factors'],
                        article_data['sentiment'],
                        article_data['url']
                    ))

                    saved_count += 1

                except Exception as e:
                    print(f"Error saving article '{article.get('title', '')}': {e}")
                    continue

            conn.commit()

        print(f"Saved {saved_count} articles to database")
        return saved_count

    def get_latest_news(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve latest news from database

        Args:
            limit: Maximum number of articles to return

        Returns:
            List of news articles
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT date, title, source, factors, sentiment, url
                    FROM news_data
                    ORDER BY date DESC, created_at DESC
                    LIMIT ?
                """, (limit,))

                articles = []
                for row in cursor.fetchall():
                    article = {
                        'date': row[0],
                        'title': row[1],
                        'source': row[2],
                        'factors': json.loads(row[3]) if row[3] else [],
                        'sentiment': row[4],
                        'url': row[5]
                    }
                    articles.append(article)

            return articles
        except Exception as e:
            print(f"Error retrieving news from database: {e}")
            return []

    def aggregate_daily_sentiment(self, days: int = 7) -> Dict[str, Any]:
        """
        Aggregate sentiment data for recent days

        Args:
            days: Number of days to look back

        Returns:
            Dictionary with daily sentiment aggregations
        """
        with sqlite3.connect(self.db_path) as conn:
            # Get sentiment data for recent days
            cursor = conn.execute("""
                SELECT date, sentiment, factors
                FROM news_data
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC
            """.format(days))

            daily_data = {}
            for row in cursor.fetchall():
                article_date = row[0]
                sentiment = row[1]
                factors = json.loads(row[2]) if row[2] else []

                if article_date not in daily_data:
                    daily_data[article_date] = {
                        'total_sentiment': 0.0,
                        'article_count': 0,
                        'factor_sentiments': {}
                    }

                daily_data[article_date]['total_sentiment'] += sentiment
                daily_data[article_date]['article_count'] += 1

                # Aggregate by factors
                for factor in factors:
                    if factor not in daily_data[article_date]['factor_sentiments']:
                        daily_data[article_date]['factor_sentiments'][factor] = {
                            'total': 0.0,
                            'count': 0
                        }

                    daily_data[article_date]['factor_sentiments'][factor]['total'] += sentiment
                    daily_data[article_date]['factor_sentiments'][factor]['count'] += 1

            # Calculate averages
            for date_key, data in daily_data.items():
                if data['article_count'] > 0:
                    data['avg_sentiment'] = round(data['total_sentiment'] / data['article_count'], 3)

                for factor, factor_data in data['factor_sentiments'].items():
                    if factor_data['count'] > 0:
                        factor_data['avg_sentiment'] = round(factor_data['total'] / factor_data['count'], 3)

        return daily_data

    def run_pipeline(self, max_results: int = 20) -> Dict[str, Any]:
        """
        Run the complete news intelligence pipeline

        Args:
            max_results: Maximum number of articles to fetch

        Returns:
            Pipeline execution results
        """
        print("Starting News Intelligence Pipeline...")

        # Fetch and process news
        articles = self.fetch_and_process_news(max_results=max_results)

        if not articles:
            return {"success": False, "message": "No articles fetched", "articles_saved": 0}

        # Save to database
        saved_count = self.save_to_database(articles)

        # Get daily sentiment aggregation
        daily_sentiment = self.aggregate_daily_sentiment(days=7)

        result = {
            "success": True,
            "articles_fetched": len(articles),
            "articles_saved": saved_count,
            "daily_sentiment": daily_sentiment,
            "timestamp": datetime.now().isoformat()
        }

        print(f"Pipeline completed successfully. Saved {saved_count} articles.")
        return result


def run_news_pipeline(max_results: int = 20) -> Dict[str, Any]:
    """
    Convenience function to run the news pipeline

    Args:
        max_results: Maximum number of articles to fetch

    Returns:
        Pipeline results
    """
    pipeline = NewsPipeline()
    return pipeline.run_pipeline(max_results=max_results)


if __name__ == "__main__":
    # Test the news pipeline
    print("Testing News Pipeline...")

    try:
        results = run_news_pipeline(max_results=5)
        print(f"Pipeline results: {json.dumps(results, indent=2)}")

        # Test retrieving latest news
        pipeline = NewsPipeline()
        latest_news = pipeline.get_latest_news(limit=3)
        print(f"\nLatest news from database:")
        for article in latest_news:
            print(f"- {article['title']} (Sentiment: {article['sentiment']})")

    except Exception as e:
        print(f"Error: {e}")