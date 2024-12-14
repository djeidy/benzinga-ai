import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from trading_agent.agent import (
    DataProcessor,
    GPT4OToolUser,
    O1PreviewReasoner,
    TradingAgent
)

@pytest.fixture
def sample_news_data():
    return [
        {
            'symbol': 'AAPL',
            'sentiment': 0.8,
            'impact': 0.9,
            'relevance': 0.7,
            'title': 'Apple Announces New Product',
            'body': 'Sample news content'
        },
        {
            'symbol': 'GOOGL',
            'sentiment': 0.6,
            'impact': 0.5,
            'relevance': 0.8,
            'title': 'Google Updates Search Algorithm',
            'body': 'Sample news content'
        }
    ]

@pytest.fixture
def sample_signals_data():
    return [
        {
            'symbol': 'AAPL',
            'sentiment': 'BULLISH',
            'cost_basis': 150000,
            'strike_price': 180.0,
            'expiration': '2024-03-15'
        },
        {
            'symbol': 'AAPL',
            'sentiment': 'BEARISH',
            'cost_basis': 100000,
            'strike_price': 170.0,
            'expiration': '2024-03-15'
        }
    ]

@pytest.fixture
def mock_benzinga_client():
    client = AsyncMock()
    client.__aenter__ = AsyncMock()
    client.__aexit__ = AsyncMock()
    client.__aenter__.return_value = client
    return client

class TestDataProcessor:
    def test_analyze_news_sentiment(self, sample_news_data):
        result = DataProcessor.analyze_news_sentiment(sample_news_data)
        assert 'sentiment' in result
        assert 'impact' in result
        assert 'relevance' in result
        assert len(result['summary']) == 2
        assert result['sentiment'] == 0.7  # (0.8 + 0.6) / 2
        assert result['impact'] == 0.7     # (0.9 + 0.5) / 2

    def test_analyze_options_activity(self, sample_signals_data):
        result = DataProcessor.analyze_options_activity(sample_signals_data)
        assert 'bullish_ratio' in result
        assert 'avg_trade_size' in result
        assert result['bullish_ratio'] == 0.5  # 1 bullish out of 2
        assert result['avg_trade_size'] == 125000  # (150000 + 100000) / 2

    def test_create_visualization(self, sample_news_data):
        result = DataProcessor.create_visualization(
            {'summary': sample_news_data}, 'sentiment_impact'
        )
        assert isinstance(result, str)
        assert 'plotly' in result.lower()

class TestGPT4OToolUser:
    @pytest.mark.asyncio
    async def test_fetch_data(self, mock_benzinga_client, sample_news_data):
        with patch('trading_agent.agent.BenzingaClient') as MockClient:
            MockClient.return_value = mock_benzinga_client
            mock_benzinga_client.get_news.return_value = sample_news_data

            tool_user = GPT4OToolUser('test_key')
            instruction = {
                'endpoint': 'news',
                'parameters': {'date': '2024-01-13'},
                'processing': {'visualize': True}
            }

            result = await tool_user.fetch_data(instruction)
            assert 'sentiment' in result
            assert 'visualization' in result
            mock_benzinga_client.get_news.assert_called_once()

    @pytest.mark.asyncio
    async def test_adjust_parameters(self):
        tool_user = GPT4OToolUser('test_key')
        instruction = {
            'type': 'news_relevance',
            'conditions': {'market_volatility': 0.7}
        }
        result = await tool_user.adjust_parameters(instruction)
        assert 'impact_score_min' in result
        assert 0.3 <= result['impact_score_min'] <= 0.9

class TestO1PreviewReasoner:
    @pytest.mark.asyncio
    async def test_analyze_market_overview(self, mock_benzinga_client, sample_news_data):
        with patch('trading_agent.agent.BenzingaClient') as MockClient:
            MockClient.return_value = mock_benzinga_client
            mock_benzinga_client.get_news.return_value = sample_news_data

            tool_user = GPT4OToolUser('test_key')
            reasoner = O1PreviewReasoner(tool_user)
            result = await reasoner.analyze_market_overview()

            assert 'news_analysis' in result
            assert 'options_activity' in result

    @pytest.mark.asyncio
    async def test_discover_trading_opportunities(self, mock_benzinga_client, sample_news_data, sample_signals_data):
        with patch('trading_agent.agent.BenzingaClient') as MockClient:
            MockClient.return_value = mock_benzinga_client
            mock_benzinga_client.get_news.return_value = sample_news_data
            mock_benzinga_client.get_signals.return_value = sample_signals_data

            tool_user = GPT4OToolUser('test_key')
            reasoner = O1PreviewReasoner(tool_user)

            # Set up initial analysis results
            reasoner.analysis_results['market_overview'] = {
                'news_analysis': {'summary': sample_news_data},
                'options_activity': {'signals': sample_signals_data}
            }

            opportunities = await reasoner.discover_trading_opportunities()
            assert isinstance(opportunities, list)
            assert all(isinstance(opp, dict) for opp in opportunities)

class TestTradingAgent:
    @pytest.mark.asyncio
    async def test_run_daily_analysis(self, mock_benzinga_client, sample_news_data, sample_signals_data):
        with patch('trading_agent.agent.BenzingaClient') as MockClient:
            MockClient.return_value = mock_benzinga_client
            mock_benzinga_client.get_news.return_value = sample_news_data
            mock_benzinga_client.get_signals.return_value = sample_signals_data

            agent = TradingAgent('test_key')
            result = await agent.run_daily_analysis()

            assert 'market_overview' in result
            assert 'trading_opportunities' in result
            assert 'optimized_parameters' in result
