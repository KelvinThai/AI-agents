import os
import json
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class NewsDataAgent:
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
        self.keywords = ["cryptocurrency", 'defi', 'nft', 'metaverse', 'web3', "blockchain"]

    def fetch_news(self):
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Fetch news from the last 7 days

        articles = []
        for keyword in self.keywords:
            response = self.newsapi.get_everything(
                q=keyword,
                from_param=start_date,
                to=end_date,
                language='en',
                sort_by='relevancy',
                page_size=10  # Limit to 10 articles per keyword
            )
            articles.extend(response['articles'])

        # Process and format the articles
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                'title': article['title'],
                'description': article['description'],
                'url': article['url'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name']
            })

        return formatted_articles

    def run(self):
        return self.fetch_news()

class NewsSummaryAgent:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "mixtral-8x7b-32768"  # A fast, relatively small model

    def summarize_news(self, news_data):
        news_prompt = f"""Summarize the following cryptocurrency news data:

{json.dumps(news_data[:10], indent=2)}
Provide a comprehensive summary that includes:
1. Top headlines (list the 5 most important)
2. Key topics and their frequency
3. Main sources
4. Emerging trends or patterns
5. Notable price movements or market events
6. Regulatory updates or government actions
7. Significant partnerships or collaborations
8. Technological advancements or innovations
9. Expert opinions or predictions

Ensure the summary is concise yet informative, highlighting the most crucial information for cryptocurrency market analysis. Include relevant statistics or data points where applicable.

Format the output as a JSON string."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant specialized in summarizing cryptocurrency-related news. Your summary will be used by a main researcher to write a comprehensive final research report. Focus on providing clear, concise, and relevant information that can be easily integrated into a larger analysis."},
                {"role": "user", "content": news_prompt}
            ],
            temperature=0.2,
            max_tokens=500
        )
        return response.choices[0].message.content

    def run(self, news_data):
        return self.summarize_news(news_data)

if __name__ == "__main__":
    # Test NewsDataAgent
    news_agent = NewsDataAgent()
    news_data = news_agent.run()
    print(f"Fetched {len(news_data)} news articles")
    for article in news_data[:5]:  # Print first 5 articles as a sample
        print(f"Title: {article['title']}")
        print(f"Source: {article['source']}")
        print(f"URL: {article['url']}")
        print("---")

    # Test NewsSummaryAgent
    summary_agent = NewsSummaryAgent()
    summary = summary_agent.run(news_data)
    print("\nSummarized News Data:")
    print(summary)
