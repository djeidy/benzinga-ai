"""GPT4O tool user for executing trading analysis tasks"""
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class GPT4OToolUser:
    """Tool user that executes instructions from O1-Preview"""

    def __init__(self, benzinga_client):
        self.client = benzinga_client
        self.current_tasks: List[Dict] = []
        self.analysis_results: Dict = {}

    async def execute_instructions(self, instructions: List[Dict]) -> Dict:
        """Execute instructions received from O1-Preview

        Args:
            instructions: List of instruction dictionaries from O1-Preview

        Returns:
            Dict containing analysis results and findings
        """
        self.current_tasks = instructions
        results = {
            'completed_tasks': [],
            'findings': {},
            'recommendations': []
        }

        for instruction in instructions:
            action = instruction.get('action')
            if action == 'research_ticker':
                result = await self._research_ticker(instruction)
                results['findings'][instruction['ticker']] = result

            elif action == 'analyze_earnings_history':
                result = await self._analyze_earnings(instruction)
                results['findings'][instruction['ticker']] = result

            elif action == 'detailed_analysis':
                result = await self._detailed_analysis(instruction)
                if result.get('confidence', 0) >= instruction.get('confidence_threshold', 0):
                    results['recommendations'].append(result)

            results['completed_tasks'].append({
                'action': action,
                'ticker': instruction.get('ticker'),
                'status': 'completed'
            })

        self.analysis_results = results
        return results

    async def _research_ticker(self, instruction: Dict) -> Dict:
        """Research a specific ticker based on instruction"""
        ticker = instruction['ticker']

        # Fetch recent news for the ticker
        news_params = {
            'tickers': ticker,
            'pagesize': 5
        }
        try:
            news_data = await self.client.get_news(news_params)
            return {
                'ticker': ticker,
                'news_count': len(news_data),
                'latest_news': [n['title'] for n in news_data[:3]],
                'action_recommended': True if len(news_data) > 0 else False
            }
        except Exception as e:
            logger.error(f"Error researching ticker {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'action_recommended': False
            }

    async def _analyze_earnings(self, instruction: Dict) -> Dict:
        """Analyze earnings history for a ticker"""
        ticker = instruction['ticker']

        # Fetch earnings calendar data
        calendar_params = {
            'tickers': ticker,
            'pagesize': 5
        }
        try:
            calendar_data = await self.client.get_calendar(calendar_params)
            earnings = calendar_data.get('earnings', [])
            return {
                'ticker': ticker,
                'earnings_events': len(earnings),
                'upcoming_earnings': [
                    {'date': e['date'], 'importance': e.get('importance', 0)}
                    for e in earnings
                ],
                'action_recommended': any(e.get('importance', 0) > 0 for e in earnings)
            }
        except Exception as e:
            logger.error(f"Error analyzing earnings for {ticker}: {str(e)}")
            return {
                'ticker': ticker,
                'error': str(e),
                'action_recommended': False
            }

    async def _detailed_analysis(self, instruction: Dict) -> Dict:
        """Perform detailed analysis of a ticker"""
        ticker = instruction['ticker']
        required_data = instruction.get('required_data', [])
        confidence_threshold = instruction.get('confidence_threshold', 0.8)

        analysis = {
            'ticker': ticker,
            'confidence': 0.0,
            'analysis_points': [],
            'historical_data': {}
        }

        # Gather all required data
        if 'recent_news_sentiment' in required_data:
            news_result = await self._research_ticker({'ticker': ticker})
            if news_result.get('action_recommended'):
                analysis['confidence'] += 0.3
                analysis['analysis_points'].append({
                    'factor': 'news_sentiment',
                    'positive': True,
                    'details': f"Found {news_result.get('news_count', 0)} recent news articles"
                })
                analysis['historical_data']['news'] = news_result

        if 'historical_price_action' in required_data:
            price_data = await self._fetch_historical_data(ticker)
            if price_data:
                price_confidence = self._analyze_price_trends(price_data)
                analysis['confidence'] += price_confidence
                analysis['analysis_points'].append({
                    'factor': 'price_trends',
                    'positive': price_confidence > 0,
                    'details': self._summarize_price_trends(price_data)
                })
                analysis['historical_data']['prices'] = price_data

        if 'trading_volume' in required_data:
            volume_data = await self._fetch_volume_data(ticker)
            if volume_data:
                volume_confidence = self._analyze_volume(volume_data)
                analysis['confidence'] += volume_confidence
                analysis['analysis_points'].append({
                    'factor': 'volume_analysis',
                    'positive': volume_confidence > 0,
                    'details': self._summarize_volume(volume_data)
                })
                analysis['historical_data']['volume'] = volume_data

        return analysis

    async def _fetch_historical_data(self, ticker: str) -> Optional[Dict]:
        """Fetch historical price data for a ticker"""
        try:
            # Use Benzinga client to fetch historical data
            # This is a placeholder until we implement the actual API call
            return {
                'ticker': ticker,
                'prices': [100.0, 101.0, 102.0],  # Placeholder data
                'dates': ['2024-01-01', '2024-01-02', '2024-01-03']
            }
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {str(e)}")
            return None

    async def _fetch_volume_data(self, ticker: str) -> Optional[Dict]:
        """Fetch trading volume data for a ticker"""
        try:
            # Use Benzinga client to fetch volume data
            # This is a placeholder until we implement the actual API call
            return {
                'ticker': ticker,
                'volume': [1000000, 1200000, 900000],  # Placeholder data
                'dates': ['2024-01-01', '2024-01-02', '2024-01-03']
            }
        except Exception as e:
            logger.error(f"Error fetching volume data for {ticker}: {str(e)}")
            return None

    def _analyze_price_trends(self, price_data: Dict) -> float:
        """Analyze price trends and return confidence adjustment"""
        prices = price_data.get('prices', [])
        if len(prices) < 2:
            return 0.0

        # Simple trend analysis (placeholder implementation)
        trend = (prices[-1] - prices[0]) / prices[0]
        if trend > 0.05:  # 5% increase
            return 0.2
        elif trend > 0.02:  # 2% increase
            return 0.1
        elif trend < -0.05:  # 5% decrease
            return -0.2
        elif trend < -0.02:  # 2% decrease
            return -0.1
        return 0.0

    def _analyze_volume(self, volume_data: Dict) -> float:
        """Analyze volume trends and return confidence adjustment"""
        volumes = volume_data.get('volume', [])
        if len(volumes) < 2:
            return 0.0

        # Simple volume analysis (placeholder implementation)
        avg_volume = sum(volumes) / len(volumes)
        latest_volume = volumes[-1]

        if latest_volume > avg_volume * 1.5:  # 50% above average
            return 0.2
        elif latest_volume > avg_volume * 1.2:  # 20% above average
            return 0.1
        elif latest_volume < avg_volume * 0.5:  # 50% below average
            return -0.2
        elif latest_volume < avg_volume * 0.8:  # 20% below average
            return -0.1
        return 0.0

    def _summarize_price_trends(self, price_data: Dict) -> str:
        """Generate a summary of price trends"""
        prices = price_data.get('prices', [])
        if len(prices) < 2:
            return "Insufficient price data"

        trend = (prices[-1] - prices[0]) / prices[0] * 100
        return f"Price {'increased' if trend > 0 else 'decreased'} by {abs(trend):.1f}% over the period"

    def _summarize_volume(self, volume_data: Dict) -> str:
        """Generate a summary of volume trends"""
        volumes = volume_data.get('volume', [])
        if len(volumes) < 2:
            return "Insufficient volume data"

        avg_volume = sum(volumes) / len(volumes)
        latest_volume = volumes[-1]
        percent_diff = (latest_volume - avg_volume) / avg_volume * 100

        return f"Latest volume is {abs(percent_diff):.1f}% {'above' if percent_diff > 0 else 'below'} average"
