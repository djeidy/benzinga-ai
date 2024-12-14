import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import pandas as pd
import plotly.graph_objects as go
from .benzinga_client import BenzingaClient

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and analyze data from Benzinga API"""

    @staticmethod
    def analyze_news_sentiment(news_data: List[Dict]) -> Dict:
        """Analyze sentiment and impact of news articles"""
        df = pd.DataFrame(news_data)
        if df.empty:
            return {'sentiment': 0, 'impact': 0, 'relevance': 0}

        # Calculate metrics
        sentiment = df.get('sentiment', 0).mean()
        impact = df.get('impact', 0).mean()
        relevance = df.get('relevance', 0).mean()

        return {
            'sentiment': sentiment,
            'impact': impact,
            'relevance': relevance,
            'summary': df.to_dict('records')
        }

    @staticmethod
    def analyze_options_activity(signals_data: List[Dict]) -> Dict:
        """Analyze options trading signals"""
        df = pd.DataFrame(signals_data)
        if df.empty:
            return {'bullish_ratio': 0, 'avg_trade_size': 0}

        bullish_ratio = len(df[df['sentiment'] == 'BULLISH']) / len(df)
        avg_trade_size = df.get('cost_basis', 0).mean()

        return {
            'bullish_ratio': bullish_ratio,
            'avg_trade_size': avg_trade_size,
            'signals': df.to_dict('records')
        }

    @staticmethod
    def create_visualization(data: Dict, viz_type: str) -> str:
        """Create visualization of analysis results"""
        if viz_type == 'sentiment_impact':
            fig = go.Figure()
            df = pd.DataFrame(data.get('summary', []))
            if not df.empty:
                fig.add_trace(go.Scatter(
                    x=df['sentiment'],
                    y=df['impact'],
                    mode='markers+text',
                    text=df['symbol'],
                    textposition='top center'
                ))
                fig.update_layout(
                    title='News Sentiment vs Impact by Stock',
                    xaxis_title='Sentiment Score',
                    yaxis_title='Impact Score'
                )
            return fig.to_html()
        return ""

class GPT4OToolUser:
    """Tool user that interacts with Benzinga API based on o1-preview's instructions"""
    def __init__(self, api_key: str):
        self.client = BenzingaClient(api_key)
        self.processor = DataProcessor()
        self.data_cache = {}

    async def fetch_data(self, instruction: Dict) -> Dict:
        """Fetch and process data based on o1-preview's instructions"""
        endpoint = instruction['endpoint']
        params = instruction.get('parameters', {})
        processing = instruction.get('processing', {})

        # Check cache first
        cache_key = f"{endpoint}:{str(params)}"
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]

        # Fetch data based on endpoint
        async with self.client as client:
            if endpoint == 'news':
                data = await client.get_news(params)
                result = self.processor.analyze_news_sentiment(data)
                if processing.get('visualize'):
                    result['visualization'] = self.processor.create_visualization(
                        result, 'sentiment_impact'
                    )
            elif endpoint == 'signals':
                data = await client.get_signals(params)
                result = self.processor.analyze_options_activity(data)
            elif endpoint == 'calendar':
                data = await client.get_calendar(params)
                result = {'events': data}
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")

        # Cache results
        self.data_cache[cache_key] = result
        return result

    async def adjust_parameters(self, instruction: Dict) -> Dict:
        """Adjust API parameters based on o1-preview's instructions"""
        adjustment_type = instruction.get('type')
        current_params = instruction.get('current_params', {})
        conditions = instruction.get('conditions', {})

        if adjustment_type == 'news_relevance':
            # Adjust news relevance threshold based on market volatility
            volatility = conditions.get('market_volatility', 0.5)
            current_params['impact_score_min'] = max(0.3, min(0.9, volatility))

        elif adjustment_type == 'options_signal':
            # Adjust options signal parameters based on market conditions
            volume = conditions.get('market_volume', 'normal')
            if volume == 'high':
                current_params['cost_basis_min'] = 200000
            else:
                current_params['cost_basis_min'] = 100000

        return current_params

class O1PreviewReasoner:
    """Reasoner that analyzes data and guides GPT4O's tool usage"""
    def __init__(self, tool_user: GPT4OToolUser):
        self.tool_user = tool_user
        self.analysis_results = {}

    async def analyze_market_overview(self) -> Dict:
        """Analyze initial market data and guide further data collection"""
        # Get pre-market news
        news_instruction = {
            'endpoint': 'news',
            'parameters': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'updated_since': (datetime.now() - timedelta(hours=12)).timestamp(),
                'channels': ['premarket', 'earnings', 'ratings'],
                'impact_score_min': 0.7
            }
        }
        news_data = await self.tool_user.fetch_data(news_instruction)

        # Get options activity
        signals_instruction = {
            'endpoint': 'signals',
            'parameters': {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'size_min': 100000
            }
        }
        signals_data = await self.tool_user.fetch_data(signals_instruction)

        # Analyze and store results
        self.analysis_results['market_overview'] = {
            'news_analysis': news_data,
            'options_activity': signals_data
        }

        return self.analysis_results['market_overview']

    async def discover_trading_opportunities(self) -> List[Dict]:
        """Discover potential trading opportunities"""
        opportunities = []

        # Analyze stocks with high news impact and options activity
        overview = self.analysis_results.get('market_overview', {})
        news = overview.get('news_analysis', {})
        options = overview.get('options_activity', {})

        # Find stocks with both high news impact and bullish options activity
        for stock in news.get('summary', []):
            if stock.get('impact', 0) > 0.8:
                # Get detailed options data for this stock
                signals_instruction = {
                    'endpoint': 'signals',
                    'parameters': {
                        'symbols': stock['symbol'],
                        'date': datetime.now().strftime('%Y-%m-%d')
                    }
                }
                stock_signals = await self.tool_user.fetch_data(signals_instruction)

                if stock_signals.get('bullish_ratio', 0) > 0.6:
                    opportunities.append({
                        'symbol': stock['symbol'],
                        'news_impact': stock['impact'],
                        'options_sentiment': stock_signals['bullish_ratio'],
                        'avg_trade_size': stock_signals['avg_trade_size']
                    })

        return opportunities

    async def optimize_parameters(self, market_conditions: Dict) -> Dict:
        """Optimize API parameters based on market conditions"""
        # Adjust news parameters based on market volatility
        news_params = await self.tool_user.adjust_parameters({
            'type': 'news_relevance',
            'conditions': {
                'market_volatility': market_conditions.get('volatility', 0.5)
            }
        })

        # Adjust options parameters based on market volume
        options_params = await self.tool_user.adjust_parameters({
            'type': 'options_signal',
            'conditions': {
                'market_volume': market_conditions.get('volume', 'normal')
            }
        })

        return {
            'news_parameters': news_params,
            'options_parameters': options_params
        }

class TradingAgent:
    """Main trading agent that coordinates o1-preview and gpt4o"""
    def __init__(self, api_key: str):
        self.tool_user = GPT4OToolUser(api_key)
        self.reasoner = O1PreviewReasoner(self.tool_user)

    async def run_daily_analysis(self):
        """Execute daily trading analysis workflow"""
        # Morning market overview
        market_overview = await self.reasoner.analyze_market_overview()

        # Discover trading opportunities
        opportunities = await self.reasoner.discover_trading_opportunities()

        # Optimize parameters based on market conditions
        params = await self.reasoner.optimize_parameters(market_overview)

        return {
            'market_overview': market_overview,
            'trading_opportunities': opportunities,
            'optimized_parameters': params
        }

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    # Load API key from environment
    load_dotenv()
    api_key = os.getenv('BENZINGA_API_KEY')

    if not api_key:
        raise ValueError("BENZINGA_API_KEY environment variable not set")

    # Initialize and run agent
    agent = TradingAgent(api_key)
    result = asyncio.run(agent.run_daily_analysis())

    # Log results
    logger.info("Daily Analysis Results:")
    logger.info(f"Market Overview: {result['market_overview']}")
    logger.info(f"Trading Opportunities: {result['trading_opportunities']}")
    logger.info(f"Optimized Parameters: {result['optimized_parameters']}")
