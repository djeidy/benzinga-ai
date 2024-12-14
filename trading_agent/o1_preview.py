"""O1-Preview reasoner for trading decisions"""
from typing import Dict, List, Optional, Set
import logging
import re

logger = logging.getLogger(__name__)

class O1PreviewReasoner:
    """Reasoner that analyzes market data and guides GPT4O's actions"""

    def __init__(self):
        self.context: Dict = {
            'analysis_parameters': {
                'sentiment_threshold': 0.55,
                'min_news_count': 2,
                'context_boost': 0.08,
                'earnings_importance_threshold': 0.6
            }
        }
        self.current_analysis: Optional[Dict] = None
        self.company_tickers = {
            'AAPL': ['apple', 'iphone'],
            'MSFT': ['microsoft'],
            'GOOGL': ['google', 'alphabet'],
            'AMZN': ['amazon'],
            'META': ['meta', 'facebook'],
            'NVDA': ['nvidia'],
            'TSLA': ['tesla'],
            'SPY': ['s&p 500', 'sp500'],
            'QQQ': ['nasdaq']
        }
        self.known_tickers: Set[str] = set(self.company_tickers.keys())

    def analyze_news(self, news_data: List[Dict]) -> Dict:
        """Analyze news data to identify trading opportunities"""
        analysis = {
            'potential_trades': [],
            'market_sentiment': {},
            'gpt4o_instructions': [],
            'analysis_parameters': self.context['analysis_parameters'].copy()
        }

        for article in news_data:
            ticker_mentions = self._extract_tickers(article)
            sentiment = self._analyze_sentiment(article)

            for ticker in ticker_mentions:
                if ticker not in analysis['market_sentiment']:
                    analysis['market_sentiment'][ticker] = []
                analysis['market_sentiment'][ticker].append(sentiment)

                if sentiment['score'] >= self.context['analysis_parameters']['sentiment_threshold']:
                    analysis['gpt4o_instructions'].append({
                        'action': 'research_ticker',
                        'ticker': ticker,
                        'reason': f"Positive sentiment in news: {article['title']}",
                        'required_confidence': self.context['analysis_parameters']['sentiment_threshold']
                    })

        for ticker, sentiments in analysis['market_sentiment'].items():
            if len(sentiments) >= self.context['analysis_parameters']['min_news_count']:
                avg_sentiment = sum(s['score'] for s in sentiments) / len(sentiments)
                if avg_sentiment >= self.context['analysis_parameters']['sentiment_threshold']:
                    analysis['potential_trades'].append({
                        'ticker': ticker,
                        'confidence': avg_sentiment,
                        'supporting_news': len(sentiments),
                        'context_boost': self.context['analysis_parameters']['context_boost']
                    })

        self.current_analysis = analysis
        return analysis

    def analyze_calendar(self, calendar_data: Dict) -> Dict:
        """Analyze calendar events for trading opportunities"""
        analysis = {
            'earnings_opportunities': [],
            'gpt4o_instructions': []
        }

        if 'earnings' in calendar_data:
            for event in calendar_data['earnings']:
                importance = event.get('importance', 0)

                if importance > 0:
                    analysis['earnings_opportunities'].append({
                        'ticker': event['ticker'],
                        'date': event['date'],
                        'importance': importance
                    })

                    analysis['gpt4o_instructions'].append({
                        'action': 'analyze_earnings_history',
                        'ticker': event['ticker'],
                        'context': f"Upcoming earnings on {event['date']}"
                    })

        return analysis

    def generate_trading_instructions(self) -> List[Dict]:
        """Generate specific instructions for GPT4O based on current analysis"""
        if not self.current_analysis:
            return []

        instructions = []

        for trade in self.current_analysis.get('potential_trades', []):
            instructions.append({
                'action': 'detailed_analysis',
                'ticker': trade['ticker'],
                'required_data': [
                    'historical_price_action',
                    'recent_news_sentiment',
                    'trading_volume'
                ],
                'confidence_threshold': 0.55
            })

        instructions.extend(self.current_analysis.get('gpt4o_instructions', []))

        return instructions

    def _extract_tickers(self, article: Dict) -> List[str]:
        """Extract valid stock tickers mentioned in article"""
        title = article.get('title', '').lower()
        content = article.get('body', '').lower()
        text = f"{title} {content}"

        potential_tickers = set(re.findall(r'\b[A-Z]{2,5}\b', article.get('title', '')))

        for ticker, names in self.company_tickers.items():
            if any(name in text for name in names):
                potential_tickers.add(ticker)

        valid_tickers = [ticker for ticker in potential_tickers
                        if ticker in self.known_tickers or
                        any(context in text.lower() for context in ['stock', 'shares', 'trading', 'price'])]

        return valid_tickers

    def _analyze_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment of news article with enhanced context awareness"""
        title = article.get('title', '').lower()
        content = article.get('body', '').lower()
        text = f"{title} {content}"

        positive_terms = {
            'surge': 0.15, 'jump': 0.15, 'rise': 0.1, 'gain': 0.1,
            'positive': 0.1, 'growth': 0.1, 'record': 0.15,
            'beat': 0.12, 'strong': 0.1, 'boost': 0.1,
            'upgrade': 0.15, 'bullish': 0.15, 'higher': 0.1,
            'success': 0.12, 'exceed': 0.12, 'above': 0.1
        }

        negative_terms = {
            'fall': -0.15, 'drop': -0.15, 'decline': -0.1,
            'negative': -0.1, 'loss': -0.12, 'weak': -0.1,
            'downgrade': -0.15, 'bearish': -0.15, 'miss': -0.12,
            'cut': -0.1, 'risk': -0.08, 'lower': -0.1,
            'below': -0.1, 'concern': -0.08
        }

        score = 0.5

        for term, weight in positive_terms.items():
            if term in text:
                score += weight

        for term, weight in negative_terms.items():
            if term in text:
                score += weight

        contexts = {
            'ai': {
                'terms': ['ai', 'artificial intelligence', 'machine learning', 'deep learning'],
                'boost': self.context['analysis_parameters']['context_boost']
            },
            'earnings': {
                'terms': ['earnings', 'revenue', 'profit', 'guidance'],
                'positive_contexts': ['beat', 'exceed', 'above', 'record', 'raise'],
                'boost': self.context['analysis_parameters']['context_boost'] * 1.5
            },
            'growth': {
                'terms': ['shipment', 'sales', 'market share', 'expansion'],
                'positive_contexts': ['record', 'increase', 'higher', 'strong'],
                'boost': self.context['analysis_parameters']['context_boost'] * 1.2
            }
        }

        for context, data in contexts.items():
            if any(term in text for term in data['terms']):
                if context == 'earnings' and any(pos in text for pos in data['positive_contexts']):
                    score += data['boost']
                elif context == 'growth' and any(pos in text for pos in data['positive_contexts']):
                    score += data['boost']
                elif context == 'ai':
                    score += data['boost']

        return {
            'score': max(0.0, min(1.0, score)),
            'title': article.get('title', ''),
            'contexts': [ctx for ctx, data in contexts.items()
                        if any(term in text for term in data['terms'])],
            'text_analyzed': len(text)
        }

    def _adjust_parameters(self, performance_data: Dict) -> None:
        """Adjust analysis parameters based on trading performance

        Args:
            performance_data: Dictionary containing performance metrics
                - success_rate: float, ratio of successful predictions
                - trades_analyzed: int, number of trades analyzed
                - avg_confidence: float, average confidence of predictions
        """
        params = self.context['analysis_parameters']
        success_rate = performance_data.get('success_rate', 0)
        trades_analyzed = performance_data.get('trades_analyzed', 0)

        # Only adjust if we have enough data
        if trades_analyzed < 5:
            return

        # Adjust sentiment threshold based on success rate
        if success_rate < 0.5:
            params['sentiment_threshold'] = min(0.8, params['sentiment_threshold'] + 0.05)
            params['min_news_count'] = min(5, params['min_news_count'] + 1)
        elif success_rate > 0.8:
            params['sentiment_threshold'] = max(0.5, params['sentiment_threshold'] - 0.02)
            params['min_news_count'] = max(1, params['min_news_count'] - 1)

        # Adjust context boost based on performance
        if success_rate < 0.4:
            params['context_boost'] = min(0.15, params['context_boost'] + 0.02)
        elif success_rate > 0.85:
            params['context_boost'] = max(0.05, params['context_boost'] - 0.01)

        # Log parameter adjustments
        logger.info(f"Adjusted parameters based on performance (success rate: {success_rate:.2f}):")
        logger.info(f"  sentiment_threshold: {params['sentiment_threshold']:.2f}")
        logger.info(f"  min_news_count: {params['min_news_count']}")
        logger.info(f"  context_boost: {params['context_boost']:.2f}")
