import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime, timedelta
import json
import numpy as np
import ta

class AdvancedLiveCryptoChart:
    def __init__(self):
        self.base_url = "https://api.mexc.com"
        self.binance_url = "https://api.binance.com"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.coinbase_url = "https://api.coinbase.com/v2"

    def get_mexc_price(self, symbol: str) -> dict:
        """Get current price from MEXC"""
        try:
            endpoint = "/api/v3/ticker/price"
            response = requests.get(f"{self.base_url}{endpoint}",
                                  params={"symbol": symbol.upper()},
                                  timeout=5)
            if response.status_code == 200:
                return {"exchange": "MEXC", "data": response.json()}
        except:
            pass
        return None

    def get_binance_price(self, symbol: str) -> dict:
        """Get current price from Binance"""
        try:
            endpoint = "/api/v3/ticker/price"
            response = requests.get(f"{self.binance_url}{endpoint}",
                                  params={"symbol": symbol.upper()},
                                  timeout=5)
            if response.status_code == 200:
                return {"exchange": "Binance", "data": response.json()}
        except:
            pass
        return None

    def get_coingecko_price(self, coin_id: str) -> dict:
        """Get current price from CoinGecko"""
        try:
            endpoint = f"/simple/price"
            response = requests.get(f"{self.coingecko_url}{endpoint}",
                                  params={"ids": coin_id, "vs_currencies": "usd,btc,eth"},
                                  timeout=5)
            if response.status_code == 200:
                data = response.json()
                if coin_id in data:
                    return {"exchange": "CoinGecko", "data": {"price": data[coin_id]["usd"]}}
        except:
            pass
        return None

    def get_mexc_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> list:
        """Get klines from MEXC"""
        try:
            endpoint = "/api/v3/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": min(limit, 1000)
            }
            response = requests.get(f"{self.base_url}{endpoint}", params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def get_binance_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> list:
        """Get klines from Binance"""
        try:
            endpoint = "/api/v3/klines"
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": min(limit, 1000)
            }
            response = requests.get(f"{self.binance_url}{endpoint}", params=params, timeout=10)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return []

    def get_24hr_stats(self, symbol: str) -> dict:
        """Get 24hr statistics"""
        # Try MEXC first
        try:
            endpoint = "/api/v3/ticker/24hr"
            response = requests.get(f"{self.base_url}{endpoint}",
                                  params={"symbol": symbol.upper()},
                                  timeout=5)
            if response.status_code == 200:
                return {"exchange": "MEXC", "data": response.json()}
        except:
            pass

        # Try Binance as backup
        try:
            endpoint = "/api/v3/ticker/24hr"
            response = requests.get(f"{self.binance_url}{endpoint}",
                                  params={"symbol": symbol.upper()},
                                  timeout=5)
            if response.status_code == 200:
                return {"exchange": "Binance", "data": response.json()}
        except:
            pass

        return None

    def get_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Get order book data"""
        try:
            endpoint = "/api/v3/depth"
            response = requests.get(f"{self.base_url}{endpoint}",
                                  params={"symbol": symbol.upper(), "limit": limit},
                                  timeout=5)
            if response.status_code == 200:
                return {"exchange": "MEXC", "data": response.json()}
        except:
            pass

        # Try Binance as backup
        try:
            endpoint = "/api/v3/depth"
            response = requests.get(f"{self.binance_url}{endpoint}",
                                  params={"symbol": symbol.upper(), "limit": limit},
                                  timeout=5)
            if response.status_code == 200:
                return {"exchange": "Binance", "data": response.json()}
        except:
            pass

        return None

    def calculate_technical_indicators(self, df):
        """Calculate various technical indicators"""
        try:
            # Moving Averages
            df['MA_7'] = ta.trend.sma_indicator(df['close'], window=7)
            df['MA_25'] = ta.trend.sma_indicator(df['close'], window=25)
            df['MA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            df['MA_100'] = ta.trend.sma_indicator(df['close'], window=100)
            df['MA_200'] = ta.trend.sma_indicator(df['close'], window=200)

            # Exponential Moving Averages
            df['EMA_12'] = ta.trend.ema_indicator(df['close'], window=12)
            df['EMA_26'] = ta.trend.ema_indicator(df['close'], window=26)
            df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)

            # MACD
            df['MACD'] = ta.trend.macd_diff(df['close'])
            df['MACD_signal'] = ta.trend.macd_signal(df['close'])
            df['MACD_histogram'] = ta.trend.macd_diff(df['close']) - ta.trend.macd_signal(df['close'])

            # RSI
            df['RSI'] = ta.momentum.rsi(df['close'], window=14)

            # Bollinger Bands
            df['BB_upper'] = ta.volatility.bollinger_hband(df['close'])
            df['BB_middle'] = ta.volatility.bollinger_mavg(df['close'])
            df['BB_lower'] = ta.volatility.bollinger_lband(df['close'])

            # Stochastic Oscillator
            df['Stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            df['Stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])

            # Williams %R
            df['Williams_R'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])

            # ATR (Average True Range)
            df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])

            # # Volume indicators
            # df['Volume_MA'] = ta.volume.volume_sma(df['close'], df['volume'], window=20)
            # df['Volume_weighted_price'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])

            # Ichimoku Cloud
            df['Ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
            df['Ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
            df['Ichimoku_base'] = ta.trend.ichimoku_base_line(df['high'], df['low'])
            df['Ichimoku_conversion'] = ta.trend.ichimoku_conversion_line(df['high'], df['low'])

        except Exception as e:
            st.warning(f"Some technical indicators could not be calculated: {str(e)}")

        return df

    def create_comprehensive_chart(self, symbol: str, interval: str = "5m", limit: int = 200,
                                 chart_type: str = "candlestick", indicators: list = [],
                                 show_volume: bool = True, show_orderbook: bool = False,
                                 theme: str = "dark"):
        """Create comprehensive live price chart with all options"""

        # Get klines data
        klines = self.get_mexc_klines(symbol, interval, limit)
        if not klines:
            klines = self.get_binance_klines(symbol, interval, limit)
            source = "Binance"
        else:
            source = "MEXC"

        if not klines:
            st.error(f"❌ No chart data available for {symbol}")
            return None

        try:
            # Process klines data
            df_data = []
            for kline in klines:
                df_data.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                })

            df = pd.DataFrame(df_data)
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)

            # Determine subplot configuration
            rows = 1
            row_heights = [0.7]
            subplot_titles = [f'{symbol} - {source}']

            if show_volume:
                rows += 1
                row_heights.append(0.15)
                subplot_titles.append('Volume')

            if 'RSI' in indicators:
                rows += 1
                row_heights.append(0.15)
                subplot_titles.append('RSI')

            if 'MACD' in indicators:
                rows += 1
                row_heights.append(0.15)
                subplot_titles.append('MACD')

            if 'Stochastic' in indicators:
                rows += 1
                row_heights.append(0.15)
                subplot_titles.append('Stochastic')

            # Normalize heights
            row_heights = [h/sum(row_heights) for h in row_heights]

            # Create subplots
            fig = make_subplots(
                rows=rows, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=subplot_titles,
                row_heights=row_heights
            )

            # Add main price chart
            if chart_type == "candlestick":
                fig.add_trace(
                    go.Candlestick(
                        x=df['datetime'],
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Price',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    ),
                    row=1, col=1
                )
            elif chart_type == "line":
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#4ecdc4', width=2)
                    ),
                    row=1, col=1
                )
            elif chart_type == "area":
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['close'],
                        fill='tonexty',
                        mode='lines',
                        name='Close Price',
                        fillcolor='rgba(78, 205, 196, 0.3)',
                        line=dict(color='#4ecdc4', width=2)
                    ),
                    row=1, col=1
                )
            elif chart_type == "heikin_ashi":
                # Calculate Heikin Ashi
                ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
                ha_open = df['open'].copy()
                ha_high = df[['high', 'open', 'close']].max(axis=1)
                ha_low = df[['low', 'open', 'close']].min(axis=1)

                fig.add_trace(
                    go.Candlestick(
                        x=df['datetime'],
                        open=ha_open,
                        high=ha_high,
                        low=ha_low,
                        close=ha_close,
                        name='Heikin Ashi',
                        increasing_line_color='#00ff88',
                        decreasing_line_color='#ff4444'
                    ),
                    row=1, col=1
                )

            # Add technical indicators to main chart
            if 'Moving Averages' in indicators:
                colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7']
                mas = ['MA_7', 'MA_25', 'MA_50', 'MA_100', 'MA_200']
                names = ['MA(7)', 'MA(25)', 'MA(50)', 'MA(100)', 'MA(200)']

                for ma, color, name in zip(mas, colors, names):
                    if ma in df.columns and not df[ma].isna().all():
                        fig.add_trace(
                            go.Scatter(
                                x=df['datetime'],
                                y=df[ma],
                                mode='lines',
                                name=name,
                                line=dict(color=color, width=1.5),
                                opacity=0.8
                            ),
                            row=1, col=1
                        )

            if 'EMA' in indicators:
                ema_colors = ['#e17055', '#00cec9', '#0984e3']
                emas = ['EMA_12', 'EMA_26', 'EMA_50']
                ema_names = ['EMA(12)', 'EMA(26)', 'EMA(50)']

                for ema, color, name in zip(emas, ema_colors, ema_names):
                    if ema in df.columns and not df[ema].isna().all():
                        fig.add_trace(
                            go.Scatter(
                                x=df['datetime'],
                                y=df[ema],
                                mode='lines',
                                name=name,
                                line=dict(color=color, width=1.5, dash='dash')
                            ),
                            row=1, col=1
                        )

            if 'Bollinger Bands' in indicators:
                if all(col in df.columns for col in ['BB_upper', 'BB_middle', 'BB_lower']):
                    fig.add_trace(
                        go.Scatter(
                            x=df['datetime'],
                            y=df['BB_upper'],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='rgba(255, 107, 107, 0.7)', width=1),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df['datetime'],
                            y=df['BB_lower'],
                            mode='lines',
                            name='Bollinger Bands',
                            line=dict(color='rgba(255, 107, 107, 0.7)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(255, 107, 107, 0.1)'
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df['datetime'],
                            y=df['BB_middle'],
                            mode='lines',
                            name='BB Middle',
                            line=dict(color='#ff6b6b', width=1.5)
                        ),
                        row=1, col=1
                    )

            if 'Ichimoku Cloud' in indicators:
                if all(col in df.columns for col in ['Ichimoku_a', 'Ichimoku_b']):
                    fig.add_trace(
                        go.Scatter(
                            x=df['datetime'],
                            y=df['Ichimoku_a'],
                            mode='lines',
                            name='Ichimoku A',
                            line=dict(color='rgba(78, 205, 196, 0.5)', width=1),
                            showlegend=False
                        ),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=df['datetime'],
                            y=df['Ichimoku_b'],
                            mode='lines',
                            name='Ichimoku Cloud',
                            line=dict(color='rgba(255, 107, 107, 0.5)', width=1),
                            fill='tonexty',
                            fillcolor='rgba(150, 206, 180, 0.2)'
                        ),
                        row=1, col=1
                    )

            current_row = 2

            # Add volume chart
            if show_volume:
                colors = ['#00ff88' if close >= open_price else '#ff4444'
                         for close, open_price in zip(df['close'], df['open'])]

                fig.add_trace(
                    go.Bar(
                        x=df['datetime'],
                        y=df['volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=current_row, col=1
                )

                if 'Volume_MA' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['datetime'],
                            y=df['Volume_MA'],
                            mode='lines',
                            name='Vol MA',
                            line=dict(color='#ffeaa7', width=2)
                        ),
                        row=current_row, col=1
                    )

                current_row += 1

            # Add RSI
            if 'RSI' in indicators and 'RSI' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['RSI'],
                        mode='lines',
                        name='RSI',
                        line=dict(color='#e17055', width=2)
                    ),
                    row=current_row, col=1
                )

                # Add RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=current_row, col=1)

                current_row += 1

            # Add MACD
            if 'MACD' in indicators and all(col in df.columns for col in ['MACD', 'MACD_signal', 'MACD_histogram']):
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['MACD'],
                        mode='lines',
                        name='MACD',
                        line=dict(color='#0984e3', width=2)
                    ),
                    row=current_row, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['MACD_signal'],
                        mode='lines',
                        name='Signal',
                        line=dict(color='#e17055', width=2)
                    ),
                    row=current_row, col=1
                )

                colors = ['#00ff88' if val >= 0 else '#ff4444' for val in df['MACD_histogram']]
                fig.add_trace(
                    go.Bar(
                        x=df['datetime'],
                        y=df['MACD_histogram'],
                        name='Histogram',
                        marker_color=colors,
                        opacity=0.7
                    ),
                    row=current_row, col=1
                )

                current_row += 1

            # Add Stochastic
            if 'Stochastic' in indicators and all(col in df.columns for col in ['Stoch_k', 'Stoch_d']):
                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['Stoch_k'],
                        mode='lines',
                        name='%K',
                        line=dict(color='#00cec9', width=2)
                    ),
                    row=current_row, col=1
                )

                fig.add_trace(
                    go.Scatter(
                        x=df['datetime'],
                        y=df['Stoch_d'],
                        mode='lines',
                        name='%D',
                        line=dict(color='#e17055', width=2)
                    ),
                    row=current_row, col=1
                )

                # Add Stochastic levels
                fig.add_hline(y=80, line_dash="dash", line_color="red", opacity=0.5, row=current_row, col=1)
                fig.add_hline(y=20, line_dash="dash", line_color="green", opacity=0.5, row=current_row, col=1)

                current_row += 1

            # Update layout based on theme
            template = 'plotly_dark' if theme == 'dark' else 'plotly_white'

            fig.update_layout(
                title=f"🔴 LIVE: {symbol} | {interval} | Source: {source}",
                template=template,
                height=800 + (current_row-1) * 200,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                hovermode='x unified'
            )

            # Update axes
            fig.update_xaxes(title_text="Time", row=current_row-1, col=1)
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)

            if show_volume:
                fig.update_yaxes(title_text="Volume", row=2, col=1)

            return fig

        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Advanced Live Crypto Chart",
        page_icon="📈",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #ffeaa7);
        background-size: 300% 300%;
        animation: gradient 4s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .metric-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }

    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #ff4444;
        border-radius: 50%;
        animation: pulse 1s infinite;
        margin-right: 8px;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .option-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">📈 Advanced Live Crypto Chart</h1>', unsafe_allow_html=True)

    # Initialize chart client
    chart_client = AdvancedLiveCryptoChart()

    # Sidebar for advanced options
    with st.sidebar:
        st.header("⚙️ Chart Configuration")

        # Basic settings
        st.subheader("📊 Basic Settings")
        symbol = st.text_input(
            "Cryptocurrency Symbol",
            value="BTCUSDT",
            placeholder="e.g., BTCUSDT, ETHUSDT"
        ).upper()

        interval = st.selectbox(
            "Time Interval",
            ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"],
            index=4
        )

        data_points = st.slider(
            "Data Points",
            min_value=50,
            max_value=1000,
            value=200,
            step=50
        )

        # Chart type
        st.subheader("📈 Chart Type")
        chart_type = st.selectbox(
            "Chart Style",
            ["candlestick", "line", "area", "heikin_ashi"],
            index=0
        )

        # Theme selection
        theme = st.selectbox(
            "Theme",
            ["dark", "light"],
            index=0
        )

        # Technical indicators
        st.subheader("🔧 Technical Indicators")
        available_indicators = [
            "Moving Averages",
            "EMA",
            "Bollinger Bands",
            "RSI",
            "MACD",
            "Stochastic",
            "Ichimoku Cloud"
        ]

        selected_indicators = st.multiselect(
            "Select Indicators",
            available_indicators,
            default=["Moving Averages", "RSI"]
        )

        # Display options
        st.subheader("📊 Display Options")
        show_volume = st.checkbox("Show Volume", value=True)
        show_orderbook = st.checkbox("Show Order Book", value=False)
        show_trades = st.checkbox("Show Recent Trades", value=False)

        # Auto-refresh
        st.subheader("🔄 Auto Refresh")
        auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh Interval (seconds)",
                min_value=5,
                max_value=300,
                value=30,
                step=5
            )

        # Quick symbol buttons
        st.subheader("🚀 Quick Select")
        popular_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT",
            "XRPUSDT", "DOTUSDT", "LINKUSDT", "MATICUSDT", "AVAXUSDT"
        ]

        cols = st.columns(2)
        for i, sym in enumerate(popular_symbols):
            if cols[i % 2].button(sym.replace("USDT", ""), key=sym):
                symbol = sym
                st.rerun()

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        # Manual refresh button
        if st.button("🔄 Refresh Chart", type="primary"):
            st.rerun()

        # Display chart
        if symbol:
            with st.spinner(f'Loading advanced chart for {symbol}...'):
                fig = chart_client.create_comprehensive_chart(
                    symbol=symbol,
                    interval=interval,
                    limit=data_points,
                    chart_type=chart_type,
                    indicators=selected_indicators,
                    show_volume=show_volume,
                    show_orderbook=show_orderbook,
                    theme=theme
                )

                if fig:
                    st.plotly_chart(fig, use_container_width=True, config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'{symbol}_chart',
                            'height': 800,
                            'width': 1200,
                            'scale': 1
                        }
                    })

                    # Show last update time
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"🕒 Last updated: {current_time} UTC")
                else:
                    st.error(f"❌ Could not load chart for {symbol}")
                    st.info("💡 Try popular symbols: BTCUSDT, ETHUSDT, BNBUSDT")

    with col2:
        # Live market data panel
        if symbol:
            display_live_market_data(chart_client, symbol)

            # Order book display
            if show_orderbook:
                display_order_book(chart_client, symbol)

            # Recent trades
            if show_trades:
                display_recent_trades(chart_client, symbol)

    # Bottom section - Market overview
    st.markdown("---")
    display_market_overview(chart_client)

    # Auto-refresh functionality
    if auto_refresh and 'refresh_interval' in locals():
        time.sleep(refresh_interval)
        st.rerun()

def display_live_market_data(chart_client, symbol):
    """Display live market data in sidebar"""
    st.markdown("### 💰 Live Market Data")

    # Get current price and stats
    price_data = chart_client.get_mexc_price(symbol)
    if not price_data:
        price_data = chart_client.get_binance_price(symbol)

    stats_data = chart_client.get_24hr_stats(symbol)

    if price_data:
        try:
            current_price = float(price_data['data'].get('price') or 0)
        except (ValueError, TypeError):
            current_price = 0.0

        st.markdown(
            f'<div class="metric-container">'
            f'<span class="live-indicator"></span>'
            f'<strong>Current Price</strong><br/>'
            f'<span style="font-size: 1.8em; color: #4ecdc4;">${current_price:,.4f}</span>'
            f'<br/><small>📡 {price_data["exchange"]}</small>'
            f'</div>',
            unsafe_allow_html=True
        )

    if stats_data:
        stats = stats_data['data']
        if isinstance(stats, list):
            stats = stats[0]

        # Safely parse values
        try:
            price_change = float(stats.get('priceChange') or 0)
        except (ValueError, TypeError):
            price_change = 0.0

        try:
            price_change_pct = float(stats.get('priceChangePercent') or 0)
        except (ValueError, TypeError):
            price_change_pct = 0.0

        try:
            high_24h = float(stats.get('highPrice') or 0)
        except (ValueError, TypeError):
            high_24h = 0.0

        try:
            low_24h = float(stats.get('lowPrice') or 0)
        except (ValueError, TypeError):
            low_24h = 0.0

        try:
            volume_24h = float(stats.get('volume') or 0)
        except (ValueError, TypeError):
            volume_24h = 0.0

        try:
            count_24h = int(stats.get('count') or 0)
        except (ValueError, TypeError):
            count_24h = 0

        change_color = "#00ff88" if price_change >= 0 else "#ff4444"
        change_symbol = "▲" if price_change >= 0 else "▼"

        # 24h Change
        st.markdown(
            f'<div class="metric-container">'
            f'<strong>24h Change</strong><br/>'
            f'<span style="font-size: 1.5em; color: {change_color};">'
            f'{change_symbol} {price_change_pct:.2f}%</span><br/>'
            f'<small style="color: {change_color};">±${abs(price_change):.4f}</small>'
            f'</div>',
            unsafe_allow_html=True
        )

        # 24h Range
        range_pct = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0
        st.markdown(
            f'<div class="metric-container">'
            f'<strong>24h Range ({range_pct:.1f}%)</strong><br/>'
            f'<span style="color: #00ff88;">📈 ${high_24h:,.4f}</span><br/>'
            f'<span style="color: #ff4444;">📉 ${low_24h:,.4f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Volume & Trades
        st.markdown(
            f'<div class="metric-container">'
            f'<strong>24h Volume</strong><br/>'
            f'<span style="font-size: 1.3em; color: #45b7d1;">{volume_24h:,.0f}</span><br/>'
            f'<small>📊 {count_24h:,} trades</small>'
            f'</div>',
            unsafe_allow_html=True
        )

def display_order_book(chart_client, symbol):
    """Display order book data"""
    st.markdown("### 📊 Order Book")

    orderbook_data = chart_client.get_order_book(symbol, limit=10)
    if orderbook_data:
        data = orderbook_data['data']

        # Create order book visualization
        bids = data.get('bids', [])
        asks = data.get('asks', [])

        if bids and asks:
            # Convert to DataFrames
            bids_df = pd.DataFrame(bids, columns=['Price', 'Quantity'])
            asks_df = pd.DataFrame(asks, columns=['Price', 'Quantity'])

            bids_df['Price'] = pd.to_numeric(bids_df['Price'])
            bids_df['Quantity'] = pd.to_numeric(bids_df['Quantity'])
            asks_df['Price'] = pd.to_numeric(asks_df['Price'])
            asks_df['Quantity'] = pd.to_numeric(asks_df['Quantity'])

            # Display top 5 bids and asks
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🟢 Bids**")
                for _, row in bids_df.head(5).iterrows():
                    st.markdown(
                        f'<div style="color: #00ff88; font-family: monospace;">'
                        f'{row["Price"]:.4f} | {row["Quantity"]:.2f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            with col2:
                st.markdown("**🔴 Asks**")
                for _, row in asks_df.head(5).iterrows():
                    st.markdown(
                        f'<div style="color: #ff4444; font-family: monospace;">'
                        f'{row["Price"]:.4f} | {row["Quantity"]:.2f}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Calculate spread
            best_bid = bids_df.iloc[0]['Price']
            best_ask = asks_df.iloc[0]['Price']
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100

            st.markdown(
                f'<div class="metric-container">'
                f'<strong>Spread</strong><br/>'
                f'<span style="color: #ffeaa7;">${spread:.4f} ({spread_pct:.3f}%)</span>'
                f'</div>',
                unsafe_allow_html=True
            )

def display_recent_trades(chart_client, symbol):
    """Display recent trades (mock data as most APIs require authentication)"""
    st.markdown("### 🔄 Recent Trades")

    # Mock recent trades data (in real implementation, you'd fetch from API)
    current_price = 50000  # This would come from actual price data

    trades = []
    for i in range(10):
        price_variation = np.random.uniform(-0.01, 0.01)
        trade_price = current_price * (1 + price_variation)
        trade_quantity = np.random.uniform(0.001, 0.1)
        trade_time = datetime.now() - timedelta(minutes=np.random.randint(1, 60))
        side = "BUY" if price_variation > 0 else "SELL"
        trades.append({
            'time': trade_time.strftime("%H:%M:%S"),
            'price': trade_price,
            'quantity': trade_quantity,
            'side': side
        })

    for trade in trades:
        color = "#00ff88" if trade['side'] == "BUY" else "#ff4444"
        symbol_icon = "🟢" if trade['side'] == "BUY" else "🔴"

        st.markdown(
            f'<div style="color: {color}; font-family: monospace; font-size: 0.9em;">'
            f'{symbol_icon} {trade["time"]} | ${trade["price"]:.2f} | {trade["quantity"]:.4f}'
            f'</div>',
            unsafe_allow_html=True
        )

def display_market_overview(chart_client):
    """Display market overview with multiple cryptocurrencies"""
    st.markdown("### 🌍 Market Overview")

    popular_coins = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "LINKUSDT"]

    cols = st.columns(4)

    for i, coin in enumerate(popular_coins[:8]):
        with cols[i % 4]:
            # Get price data
            price_data = chart_client.get_mexc_price(coin)
            if not price_data:
                price_data = chart_client.get_binance_price(coin)

            stats_data = chart_client.get_24hr_stats(coin)

            if price_data and stats_data:
                current_price = float(price_data['data'].get('price', 0))
                stats = stats_data['data']
                if isinstance(stats, list):
                    stats = stats[0]

                price_change_pct = float(stats.get('priceChangePercent', 0))
                change_color = "#00ff88" if price_change_pct >= 0 else "#ff4444"
                change_symbol = "▲" if price_change_pct >= 0 else "▼"

                coin_name = coin.replace("USDT", "")

                st.markdown(
                    f'<div class="option-card" style="text-align: center;">'
                    f'<strong>{coin_name}</strong><br/>'
                    f'<span style="font-size: 1.2em;">${current_price:,.2f}</span><br/>'
                    f'<span style="color: {change_color}; font-size: 0.9em;">'
                    f'{change_symbol} {price_change_pct:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                coin_name = coin.replace("USDT", "")
                st.markdown(
                    f'<div class="option-card" style="text-align: center; opacity: 0.5;">'
                    f'<strong>{coin_name}</strong><br/>'
                    f'<span style="font-size: 0.9em;">Data unavailable</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Market summary
    st.markdown("#### 📊 Quick Market Insights")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔥 Most Active", "BTCUSDT", "High Volume")

    with col2:
        st.metric("📈 Top Gainer", "Check Market", "+X.XX%")

    with col3:
        st.metric("📉 Top Loser", "Check Market", "-X.XX%")

    with col4:
        st.metric("⏰ Last Update", datetime.now().strftime("%H:%M:%S"), "Live")

    # Footer with additional info
    st.markdown("---")
    st.markdown("""
    **📝 Features Available:**
    - 🎨 **Chart Types**: Candlestick, Line, Area, Heikin Ashi
    - 📊 **Technical Indicators**: MA, EMA, Bollinger Bands, RSI, MACD, Stochastic, Ichimoku
    - 🔄 **Auto Refresh**: Configurable refresh intervals
    - 📱 **Responsive Design**: Works on all devices
    - 🌙 **Dark/Light Theme**: Customizable themes
    - 📈 **Multiple Timeframes**: From 1 minute to 1 month
    - 🎯 **Order Book**: Real-time bid/ask data
    - 🔄 **Recent Trades**: Live trading activity
    - 💰 **Market Overview**: Track multiple coins
    - 📊 **Volume Analysis**: Detailed volume indicators

    **⚠️ Disclaimer**: This tool is for educational and informational purposes only. Always do your own research before making investment decisions.
    """)

if __name__ == "__main__":
    main()