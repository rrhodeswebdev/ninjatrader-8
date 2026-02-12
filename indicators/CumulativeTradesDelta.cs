using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
using SharpDX;
using SharpDX.Direct2D1;

namespace NinjaTrader.NinjaScript.Indicators.WolfToolz
{
	/// Reset mode values: 0 = None, 1 = ETH only, 2 = ETH + RTH (default)

	public class CumulativeTradesDelta : Indicator
	{
		#region Private Fields

		private double currentAskTrades;
		private double currentBidTrades;
		private double currentDelta;
		private double previousClose;
		private double lastTradePrice;
		private double currentAsk;
		private double currentBid;
		private int lastBarIndex;
		private double deltaHigh;
		private double deltaLow;

		private Series<double> ctdOpen;
		private Series<double> ctdHigh;
		private Series<double> ctdLow;
		private Series<double> ctdClose;

		private SessionIterator sessionIterator;
		private SessionIterator rthSessionIterator;
		private DateTime currentSessionBegin;
		private DateTime currentTradingDay;
		private DateTime currentRthSessionBegin;
		private bool rthResetDone;

		#endregion

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description					= "Cumulative Trades Delta";
				Name						= "CumulativeTradesDelta";
				Calculate					= Calculate.OnEachTick;
				IsOverlay					= false;
				DisplayInDataBox			= true;
				DrawOnPricePanel			= false;
				IsSuspendedWhileInactive	= true;
				MaximumBarsLookBack			= MaximumBarsLookBack.Infinite;

				ResetMode					= 2;
				RthSessionTemplate			= "CME US Index Futures RTH";
				BullishColor				= Brushes.Magenta;
				BearishColor				= Brushes.Magenta;
				BarThickness				= 2;

				AddPlot(new Stroke(Brushes.Transparent, 0), PlotStyle.Line, "CTD");
			}
			else if (State == State.Configure)
			{
				AddDataSeries(BarsPeriodType.Tick, 1);
				AddDataSeries(Instrument.FullName, new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1440 }, RthSessionTemplate);
			}
			else if (State == State.DataLoaded)
			{
				currentAskTrades	= 0;
				currentBidTrades	= 0;
				currentDelta		= 0;
				previousClose		= 0;
				lastTradePrice		= 0;
				currentAsk			= 0;
				currentBid			= 0;
				lastBarIndex		= -1;
				deltaHigh			= double.MinValue;
				deltaLow			= double.MaxValue;

				ctdOpen		= new Series<double>(this);
				ctdHigh		= new Series<double>(this);
				ctdLow		= new Series<double>(this);
				ctdClose	= new Series<double>(this);

				sessionIterator			= new SessionIterator(Bars);
				rthSessionIterator		= new SessionIterator(BarsArray[2]);
				currentSessionBegin		= DateTime.MinValue;
				currentTradingDay		= DateTime.MinValue;
				currentRthSessionBegin	= DateTime.MinValue;
				rthResetDone			= false;
			}
		}

		protected override void OnMarketData(MarketDataEventArgs marketDataUpdate)
		{
			if (marketDataUpdate.MarketDataType == MarketDataType.Ask)
				currentAsk = marketDataUpdate.Price;
			else if (marketDataUpdate.MarketDataType == MarketDataType.Bid)
				currentBid = marketDataUpdate.Price;
		}

		protected override void OnBarUpdate()
		{
			if (BarsInProgress == 1)
			{
				if (CurrentBars[0] < 0 || CurrentBars[1] < 0)
					return;

				int primaryBar = CurrentBars[0];

				// Detect bar transition on the primary series
				if (primaryBar != lastBarIndex)
				{
					if (lastBarIndex >= 0)
						previousClose += currentDelta;

					currentAskTrades	= 0;
					currentBidTrades	= 0;
					currentDelta		= 0;
					deltaHigh			= double.MinValue;
					deltaLow			= double.MaxValue;

					lastBarIndex = primaryBar;
				}

				double tradePrice = Closes[1][0];

				// Classify: bid/ask when available, uptick/downtick as fallback
				// Count 1 trade instead of accumulating volume
				if (currentAsk > 0 && currentBid > 0)
				{
					if (tradePrice >= currentAsk)
						currentAskTrades += 1;
					else if (tradePrice <= currentBid)
						currentBidTrades += 1;
					else
					{
						// Trade between bid and ask - use uptick/downtick
						if (lastTradePrice > 0 && tradePrice > lastTradePrice)
							currentAskTrades += 1;
						else if (lastTradePrice > 0 && tradePrice < lastTradePrice)
							currentBidTrades += 1;
					}
				}
				else
				{
					// No bid/ask data - fall back to pure uptick/downtick
					if (lastTradePrice > 0)
					{
						if (tradePrice > lastTradePrice)
							currentAskTrades += 1;
						else if (tradePrice < lastTradePrice)
							currentBidTrades += 1;
					}
				}

				lastTradePrice = tradePrice;
				currentDelta = currentAskTrades - currentBidTrades;

				// Track bar-relative high/low of delta (cumulative offset applied in BIP==0)
				deltaHigh = Math.Max(deltaHigh, currentDelta);
				deltaLow = Math.Min(deltaLow, currentDelta);
			}
			else if (BarsInProgress == 0)
			{
				if (CurrentBar < 0)
					return;

				// Detect session breaks using SessionIterator
				if (Bars.IsFirstBarOfSession)
				{
					sessionIterator.GetNextSession(Time[0], true);
					DateTime tradingDay = sessionIterator.ActualTradingDayExchange;

					// ETH reset: new trading day detected
					if (tradingDay != currentTradingDay)
					{
						if (currentTradingDay != DateTime.MinValue && ResetMode >= 1)
							previousClose = 0;

						currentTradingDay = tradingDay;
						rthResetDone = false;
					}

					currentSessionBegin = sessionIterator.ActualSessionBegin;

					// Update RTH session begin time for time-based detection
					if (ResetMode == 2)
					{
						rthSessionIterator.GetNextSession(Time[0], true);
						currentRthSessionBegin = rthSessionIterator.ActualSessionBegin;
					}
				}

				// RTH reset: check if bar time has crossed RTH session begin
				if (ResetMode == 2 && !rthResetDone && currentRthSessionBegin != DateTime.MinValue && Time[0] >= currentRthSessionBegin)
				{
					previousClose = 0;
					rthResetDone = true;
				}

				if (CurrentBar == 0)
					previousClose = 0;

				double close = previousClose + currentDelta;
				double high  = deltaHigh == double.MinValue ? close : previousClose + deltaHigh;
				double low   = deltaLow == double.MaxValue ? close : previousClose + deltaLow;
				double open  = Math.Min(Math.Max(previousClose, low), high);

				ctdOpen[0]	= open;
				ctdHigh[0]	= high;
				ctdLow[0]	= low;
				ctdClose[0]	= close;
				Value[0]	= close;
			}
		}

		public override void OnCalculateMinMax()
		{
			MinValue = double.MaxValue;
			MaxValue = double.MinValue;

			for (int index = ChartBars.FromIndex; index <= ChartBars.ToIndex; index++)
			{
				double high = ctdHigh.GetValueAt(index);
				double low  = ctdLow.GetValueAt(index);

				MaxValue = Math.Max(MaxValue, high);
				MinValue = Math.Min(MinValue, low);
			}
		}

		protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
		{
			if (ChartBars == null || ctdOpen == null)
				return;

			float barWidth = (float)chartControl.BarWidth;

			for (int index = ChartBars.FromIndex; index <= ChartBars.ToIndex; index++)
			{
				double open  = ctdOpen.GetValueAt(index);
				double high  = ctdHigh.GetValueAt(index);
				double low   = ctdLow.GetValueAt(index);
				double close = ctdClose.GetValueAt(index);

				float x      = chartControl.GetXByBarIndex(ChartBars, index);
				float yOpen  = chartScale.GetYByValue(open);
				float yHigh  = chartScale.GetYByValue(high);
				float yLow   = chartScale.GetYByValue(low);
				float yClose = chartScale.GetYByValue(close);

				bool bullish = close >= open;
				System.Windows.Media.Brush wpfBrush = bullish ? BullishColor : BearishColor;

				using (var brush = wpfBrush.ToDxBrush(RenderTarget))
				{
					float thickness = (float)BarThickness;

					// Draw vertical line (high to low)
					RenderTarget.DrawLine(
						new SharpDX.Vector2(x, yHigh),
						new SharpDX.Vector2(x, yLow),
						brush, thickness);

					// Draw open tick (left)
					RenderTarget.DrawLine(
						new SharpDX.Vector2(x - barWidth, yOpen),
						new SharpDX.Vector2(x, yOpen),
						brush, thickness);

					// Draw close tick (right)
					RenderTarget.DrawLine(
						new SharpDX.Vector2(x, yClose),
						new SharpDX.Vector2(x + barWidth, yClose),
						brush, thickness);
				}
			}
		}

		#region Properties

		[NinjaScriptProperty]
		[Range(0, 2)]
		[Display(Name = "Reset Mode", Description = "0 = None (never reset), 1 = ETH (reset at ETH open only), 2 = ETH+RTH (reset at both ETH and RTH open)", Order = 1, GroupName = "Parameters")]
		public int ResetMode { get; set; }

		[XmlIgnore]
		[Display(Name = "Bullish Color", Description = "Color for bullish (up) bars", Order = 2, GroupName = "Parameters")]
		public System.Windows.Media.Brush BullishColor { get; set; }

		[Browsable(false)]
		public string BullishColorSerialize
		{
			get { return Serialize.BrushToString(BullishColor); }
			set { BullishColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Bearish Color", Description = "Color for bearish (down) bars", Order = 3, GroupName = "Parameters")]
		public System.Windows.Media.Brush BearishColor { get; set; }

		[Browsable(false)]
		public string BearishColorSerialize
		{
			get { return Serialize.BrushToString(BearishColor); }
			set { BearishColor = Serialize.StringToBrush(value); }
		}

		[NinjaScriptProperty]
		[Display(Name = "RTH Session Template", Description = "Name of the RTH trading hours template for RTH session reset detection", Order = 5, GroupName = "Parameters")]
		public string RthSessionTemplate { get; set; }

		[NinjaScriptProperty]
		[Range(1, 10)]
		[Display(Name = "Bar Thickness", Description = "Thickness of the OHLC bar lines", Order = 6, GroupName = "Parameters")]
		public int BarThickness { get; set; }

		#endregion
	}
}
