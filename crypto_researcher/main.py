import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import json
import logging
from pycoingecko import CoinGeckoAPI
from dotenv import load_dotenv
from groq import Groq
import os

# Load environment variables
load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

print("Initialized Groq client")

class CryptoResearchAgent:
    def __init__(self):
        """Initialize the crypto research agent with necessary tools and APIs"""
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing CryptoResearchAgent")
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Configure sources and symbols
        self.news_sources = [
            'https://cointelegraph.com/',
            'https://www.coindesk.com/',
            'https://cryptonews.com/',
            'https://decrypt.co/',
            'https://www.theblockcrypto.com/'
        ]
        self.crypto_symbols = ['BTC', 'ETH', 'SOL']
        
        self.cg = CoinGeckoAPI()
        self.trend_categories = ['defi', 'nft', 'metaverse', 'web3', 'layer-2']
        self.num_trending = 10  # Number of trending coins to fetch per category
        
        self.logger.info("Initializing trending crypto data")
        self.trending_cryptos = self._get_trending_cryptos()
        
        self.logger.info("CryptoResearchAgent initialized successfully")
        
    def fetch_news(self, sources):
        """Fetch news from multiple crypto news sources"""
        self.logger.info(f"Fetching news from {len(sources)} sources")
        all_articles = []
        
        for source in sources:
            try:
                self.logger.info(f"Fetching news from: {source}")
                response = requests.get(source, headers=self.headers)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract articles (implementation would depend on specific website structure)
                articles = self._parse_articles(soup)
                self.logger.info(f"Parsed {len(articles)} articles from {source}")
                all_articles.extend(articles)
            except Exception as e:
                self.logger.error(f"Error fetching from {source}: {str(e)}")
                
        self.logger.info(f"Total articles fetched: {len(all_articles)}")
        return all_articles
    
    def _parse_articles(self, soup):
        """Parse HTML to extract article information using BeautifulSoup"""
        self.logger.info("Parsing articles from HTML using BeautifulSoup")
        articles = []
        for article in soup.find_all('article'):
            try:
                title = article.find('h2').text.strip() if article.find('h2') else "No title"
                summary = article.find('p').text.strip() if article.find('p') else "No summary"
                link = article.find('a')['href'] if article.find('a') else "#"
                
                parsed_data = {
                    'title': title,
                    'summary': summary,
                    'link': link
                }
                articles.append(parsed_data)
                self.logger.debug(f"Parsed article: {parsed_data['title']}")
            except Exception as e:
                self.logger.warning(f"Failed to parse an article: {str(e)}")
                continue
        
        self.logger.info(f"Parsed {len(articles)} articles")
        return articles
    
    def fetch_market_data(self, symbols, period='7d'):
        """Fetch market data for specified crypto symbols"""
        self.logger.info(f"Fetching market data for {len(symbols)} symbols")
        market_data = {}
        
        for symbol in symbols:
            try:
                self.logger.info(f"Fetching data for {symbol}")
                ticker = yf.Ticker(f"{symbol}-USD")
                data = ticker.history(period=period)
                market_data[symbol] = {
                    'price_data': data['Close'].to_dict(),
                    'volume': data['Volume'].to_dict(),
                    'price_change': ((data['Close'][-1] - data['Close'][0]) / data['Close'][0] * 100)
                }
                self.logger.debug(f"Fetched data for {symbol}: Price change {market_data[symbol]['price_change']:.2f}%")
            except Exception as e:
                self.logger.error(f"Error fetching market data for {symbol}: {str(e)}")
                
        self.logger.info(f"Market data fetched for {len(market_data)} symbols")
        return market_data
    
    def generate_report(self, articles, market_data):
        """Generate a comprehensive research report using Groq"""
        self.logger.info("Generating comprehensive research report using Groq")
        
        # Prepare input data for the AI model
        input_data = {
            'articles': articles[:10],  # Top 10 most recent articles
            'market_data': market_data,
        }
        
        # Construct the prompt for the AI model
        prompt = f"""
        Generate a comprehensive crypto research report based on the following data:
        
        1. Recent News Articles: {input_data['articles']}
        2. Market Data: {input_data['market_data']}
        
        The report should include:
        1. An executive summary of key findings
        2. Analysis of recent news and their potential impact
        3. Market trends and patterns
        4. Identification of trending topics in the crypto space
        5. Potential impacts on major cryptocurrencies (BTC, ETH, SOL)
        
        Format the report in a clear, professional structure with headings and subheadings.
        Make it a detailed 4 5 pages report.
        """
        
        # Generate the report using Groq
        try:
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a professional crypto analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7,
            )
            
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'content': completion.choices[0].message.content,
            }
            
            self.logger.info("Research report generated successfully using Groq")
        except Exception as e:
            self.logger.error(f"Error generating report with Groq: {str(e)}")
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'error': f"Failed to generate report: {str(e)}",
            }
        
        return report
    
    def _generate_summary(self, articles, market_data):
        """Generate an executive summary of findings"""
        # Implementation would include key insights from both news and market data
        pass
    
    def _extract_trending_topics(self, articles):
        """Extract common themes and trending topics from articles using Groq"""
        self.logger.info("Extracting trending topics from articles using Groq")
        
        # Prepare the input data
        article_texts = [f"Title: {article['title']}\nSummary: {article['summary']}" for article in articles[:10]]  # Limit to 10 articles to avoid token limits
        input_text = "\n\n".join(article_texts)
        
        prompt = f"""
        Analyze the following crypto news articles and extract the top 5 trending topics or themes. 
        For each topic, provide a brief explanation of its significance in the current crypto landscape.

        Articles:
        {input_text}

        Please format your response as a JSON object with the following structure:
        {{
            "trending_topics": [
                {{
                    "topic": "Name of the topic",
                    "explanation": "Brief explanation of the topic's significance"
                }},
                ...
            ]
        }}
        """

        try:
            completion = client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": "You are a crypto market analyst specializing in identifying trends from news articles."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.5,
            )
            
            trending_topics = json.loads(completion.choices[0].message.content)
            self.logger.info(f"Successfully extracted {len(trending_topics['trending_topics'])} trending topics")
            return trending_topics
        except Exception as e:
            self.logger.error(f"Error extracting trending topics: {str(e)}")
            return {"trending_topics": []}
    
    def _get_trending_cryptos(self):
        """Fetch trending cryptocurrencies from CoinGecko"""
        self.logger.info("Fetching trending cryptocurrencies")
        trending_cryptos = {}
        
        try:
            for category in self.trend_categories:
                trending_cryptos[category] = self.cg.get_search_trending(category)['coins'][:self.num_trending]
            self.logger.info(f"Fetched trending cryptos for {len(self.trend_categories)} categories")
        except Exception as e:
            self.logger.error(f"Error fetching trending cryptos: {str(e)}")
        
        return trending_cryptos

    def run_research(self):
        """Run the entire research process"""
        self.logger.info("Starting crypto research process")

        # Fetch data
        self.logger.info("Fetching news articles")
        articles = self.fetch_news(self.news_sources)
        self.logger.info("Fetching market data")
        market_data = self.fetch_market_data(self.crypto_symbols)

        # Generate report
        self.logger.info("Generating final report")
        report = self.generate_report(articles, market_data)

        # Save report
        report_filename = f'crypto_report_{datetime.now().strftime("%Y%m%d")}.json'
        self.logger.info(f"Saving report to {report_filename}")
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=4)

        self.logger.info("Crypto research process completed")

def main():
    agent = CryptoResearchAgent()
    agent.run_research()

if __name__ == "__main__":
    main()
