"""
News NLP Module for Gold Silver AI
Handles factor classification and sentiment analysis for news articles
"""

import re
from typing import List, Dict, Any
import json
from datetime import datetime

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


# Factor mapping with keywords
FACTORS = {
    "inflation": ["inflation", "cpi", "price rise", "cost of living", "consumer price"],
    "interest_rate": ["repo rate", "interest rate", "fed", "rbi", "monetary policy", "rate hike", "rate cut"],
    "currency": ["usd", "dollar", "rupee", "usd/inr", "exchange rate", "forex", "currency"],
    "oil": ["crude oil", "oil price", "petroleum", "brent crude", "wti"],
    "geopolitics": ["war", "conflict", "crisis", "tension", "sanctions", "diplomatic"],
    "gold_market": ["gold", "silver", "bullion", "precious metals", "gold price", "silver price"]
}


class NewsNLP:
    """Handles NLP processing for news articles"""

    def __init__(self):
        if TextBlob is None:
            raise ImportError("TextBlob package is required. Install with: pip install textblob")

        try:
            # Test TextBlob initialization
            test_blob = TextBlob("test")
            _ = test_blob.sentiment
        except Exception as e:
            raise ImportError(f"Failed to initialize TextBlob: {e}")

    def classify_factors(self, text: str) -> List[str]:
        """
        Classify text into relevant factors based on keywords

        Args:
            text: Input text to classify

        Returns:
            List of detected factors
        """
        if not text:
            return []

        text_lower = text.lower()
        detected_factors = []

        for factor, keywords in FACTORS.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    detected_factors.append(factor)
                    break  # Only add factor once

        # Remove duplicates while preserving order
        seen = set()
        unique_factors = []
        for factor in detected_factors:
            if factor not in seen:
                unique_factors.append(factor)
                seen.add(factor)

        return unique_factors

    def get_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using TextBlob

        Args:
            text: Input text to analyze

        Returns:
            Sentiment polarity score (-1 to 1)
        """
        if not text:
            return 0.0

        try:
            blob = TextBlob(text)
            return round(blob.sentiment.polarity, 3)
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0.0

    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single article with NLP analysis

        Args:
            article: Raw article dictionary

        Returns:
            Processed article with NLP results
        """
        # Combine title and description for analysis
        full_text = f"{article.get('title', '')} {article.get('description', '')}"

        # Classify factors
        factors = self.classify_factors(full_text)

        # Get sentiment
        sentiment = self.get_sentiment(full_text)

        # Create processed article
        processed_article = {
            "title": article.get('title', ''),
            "source": article.get('source', ''),
            "date": article.get('published_date', ''),
            "factors": factors,
            "sentiment": sentiment,
            "url": article.get('url', '')
        }

        return processed_article

    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple articles

        Args:
            articles: List of raw articles

        Returns:
            List of processed articles
        """
        processed_articles = []
        for article in articles:
            processed = self.process_article(article)
            processed_articles.append(processed)

        return processed_articles

    def filter_relevant_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter articles to keep only those relevant to gold/silver/macro factors

        Args:
            articles: List of processed articles

        Returns:
            Filtered list of relevant articles
        """
        relevant_keywords = ["gold", "silver", "inflation", "rbi", "usd", "oil", "interest rate", "crude oil"]

        filtered_articles = []
        for article in articles:
            text = f"{article['title']} {article.get('description', '')}".lower()

            # Check if article contains relevant keywords
            is_relevant = any(keyword in text for keyword in relevant_keywords)

            if is_relevant:
                filtered_articles.append(article)

        return filtered_articles

    def aggregate_sentiment(self, articles: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate sentiment by factors

        Args:
            articles: List of processed articles

        Returns:
            Dictionary with average sentiment per factor
        """
        factor_sentiments = {}
        factor_counts = {}

        for article in articles:
            sentiment = article.get('sentiment', 0.0)
            factors = article.get('factors', [])

            for factor in factors:
                if factor not in factor_sentiments:
                    factor_sentiments[factor] = 0.0
                    factor_counts[factor] = 0

                factor_sentiments[factor] += sentiment
                factor_counts[factor] += 1

        # Calculate averages
        aggregated = {}
        for factor in factor_sentiments:
            if factor_counts[factor] > 0:
                aggregated[factor] = round(factor_sentiments[factor] / factor_counts[factor], 3)

        return aggregated


# Convenience functions
def classify_factors(text: str) -> List[str]:
    """Convenience function for factor classification"""
    nlp = NewsNLP()
    return nlp.classify_factors(text)


def get_sentiment(text: str) -> float:
    """Convenience function for sentiment analysis"""
    nlp = NewsNLP()
    return nlp.get_sentiment(text)


def process_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for processing single article"""
    nlp = NewsNLP()
    return nlp.process_article(article)


if __name__ == "__main__":
    # Test the NLP module
    print("Testing News NLP Module...")

    test_text = "Gold prices rise as inflation concerns grow and RBI considers interest rate hike"

    try:
        # Test factor classification
        factors = classify_factors(test_text)
        print(f"Detected factors: {factors}")

        # Test sentiment analysis
        sentiment = get_sentiment(test_text)
        print(f"Sentiment score: {sentiment}")

        # Test article processing
        test_article = {
            "title": "Gold Prices Surge Amid Inflation Fears",
            "description": "Gold prices hit new highs as investors seek safe haven assets due to rising inflation and geopolitical tensions.",
            "published_date": "2024-01-15",
            "source": "Economic Times",
            "url": "https://example.com"
        }

        processed = process_article(test_article)
        print(f"\nProcessed article: {json.dumps(processed, indent=2)}")

    except Exception as e:
        print(f"Error: {e}")