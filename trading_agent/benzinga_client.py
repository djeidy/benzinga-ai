import aiohttp
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class BenzingaClient:
    """Async HTTP client for Benzinga API"""

    def __init__(self, api_key: str, base_url: str = "https://api.benzinga.com/api"):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make authenticated request to Benzinga API"""
        if not self.session:
            raise RuntimeError("Client session not initialized. Use async with context.")

        params['token'] = self.api_key
        url = f"{self.base_url}{endpoint}"

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"API request failed: {str(e)}")
            raise

    async def get_news(self, params: Dict) -> Dict:
        """Fetch news data from Benzinga API"""
        return await self._make_request('/v2/news', params)

    async def get_signals(self, params: Dict) -> Dict:
        """Fetch options signals data"""
        return await self._make_request('/v1/signals', params)

    async def get_calendar(self, params: Dict) -> Dict:
        """Fetch calendar events data"""
        return await self._make_request('/v1/calendar', params)

    async def get_bars(self, params: Dict) -> Dict:
        """Fetch historical price bars"""
        return await self._make_request('/v1/history', params)
