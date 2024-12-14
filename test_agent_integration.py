"""Integration test for the complete trading agent system"""
import asyncio
import logging
from trading_agent.simple_client import SimpleBenzingaClient
from trading_agent.o1_preview import O1PreviewReasoner
from trading_agent.gpt4o_tool import GPT4OToolUser
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv('BENZINGA_API_KEY')

async def test_trading_agent():
    """Test the complete trading agent system with performance tracking"""
    async with SimpleBenzingaClient(API_KEY) as client:
        # Initialize components
        reasoner = O1PreviewReasoner()
        tool_user = GPT4OToolUser(client)

        # Initialize performance metrics
        performance_metrics = {
            'trades_analyzed': 0,
            'successful_predictions': 0,
            'parameter_adjustments': [],
            'iterations': []
        }

        # Run multiple test iterations
        for iteration in range(5):
            logger.info(f"\nStarting iteration {iteration + 1}/5")

            # Test news analysis
            yesterday = int((datetime.now() - timedelta(days=1)).timestamp())
            news_params = {
                'tickers': 'AAPL,MSFT,GOOGL,NVDA,META',  # Extended ticker list
                'pagesize': 15,
                'updatedSince': yesterday
            }

            try:
                # Get news data
                news_data = await client.get_news(news_params)
                logger.info(f"Retrieved {len(news_data)} news articles")

                # Let O1-Preview analyze the news
                news_analysis = reasoner.analyze_news(news_data)
                logger.info("\nO1-Preview Analysis:")
                logger.info(f"Potential trades: {len(news_analysis['potential_trades'])}")
                logger.info(f"Market sentiment tracked: {len(news_analysis['market_sentiment'])}")

                # Generate instructions for GPT4O
                instructions = reasoner.generate_trading_instructions()
                logger.info(f"\nGenerated {len(instructions)} instructions for GPT4O")

                # Let GPT4O execute the instructions
                if instructions:
                    results = await tool_user.execute_instructions(instructions)
                    logger.info("\nGPT4O Results:")
                    logger.info(f"Completed tasks: {len(results['completed_tasks'])}")
                    logger.info(f"Generated recommendations: {len(results['recommendations'])}")

                    # Track performance metrics
                    iteration_metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'trades_analyzed': len(results['recommendations']),
                        'high_confidence_predictions': len([r for r in results['recommendations']
                                                         if r['confidence'] > 0.7]),
                        'parameters': news_analysis['analysis_parameters']
                    }

                    performance_metrics['trades_analyzed'] += iteration_metrics['trades_analyzed']
                    performance_metrics['successful_predictions'] += iteration_metrics['high_confidence_predictions']
                    performance_metrics['iterations'].append(iteration_metrics)

                    # Calculate success rate and update reasoner
                    if iteration_metrics['trades_analyzed'] > 0:
                        success_rate = (iteration_metrics['high_confidence_predictions'] /
                                      iteration_metrics['trades_analyzed'])
                        reasoner.context['last_performance'] = {
                            'success_rate': success_rate,
                            'iteration': iteration,
                            'trades_analyzed': iteration_metrics['trades_analyzed']
                        }
                        logger.info(f"\nIteration {iteration + 1} success rate: {success_rate:.2f}")

                    # Track parameter adjustments
                    if iteration > 0:
                        prev_params = performance_metrics['iterations'][iteration - 1]['parameters']
                        curr_params = iteration_metrics['parameters']
                        if prev_params != curr_params:
                            performance_metrics['parameter_adjustments'].append({
                                'iteration': iteration,
                                'previous': prev_params,
                                'current': curr_params
                            })
                else:
                    logger.info("No trading instructions generated")

            except Exception as e:
                logger.error(f"Error in iteration {iteration + 1}: {str(e)}")
                continue

            # Add delay between iterations
            if iteration < 4:  # Don't wait after the last iteration
                await asyncio.sleep(2)  # 2-second delay between iterations

        # Log final performance metrics
        logger.info("\nFinal Performance Metrics:")
        logger.info(f"Total trades analyzed: {performance_metrics['trades_analyzed']}")
        logger.info(f"Total successful predictions: {performance_metrics['successful_predictions']}")
        logger.info(f"Overall success rate: {performance_metrics['successful_predictions'] / max(1, performance_metrics['trades_analyzed']):.2f}")
        logger.info(f"Parameter adjustments made: {len(performance_metrics['parameter_adjustments'])}")

        # Save metrics to file
        with open('performance_metrics.json', 'w') as f:
            json.dump(performance_metrics, f, indent=2)

if __name__ == "__main__":
    asyncio.run(test_trading_agent())
