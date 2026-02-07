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
	public class CumulativeVolumeDelta : Indicator
	{
		#region Private Fields

		private double currentAskVolume;
		private double currentBidVolume;
		private double currentDelta;
		private double previousClose;
		private double lastTradePrice;
		private double currentAsk;
		private double currentBid;
		private int lastBarIndex;
		private double deltaHigh;
		private double deltaLow;

		private Series<double> cvdOpen;
		private Series<double> cvdHigh;
		private Series<double> cvdLow;
		private Series<double> cvdClose;

		private SessionIterator sessionIterator;
		private DateTime currentSessionBegin;

		#endregion

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description					= "Cumulative Volume Delta";
				Name						= "CumulativeVolumeDelta";
				Calculate					= Calculate.OnEachTick;
				IsOverlay					= false;
				DisplayInDataBox			= true;
				DrawOnPricePanel			= false;
				IsSuspendedWhileInactive	= true;
				MaximumBarsLookBack			= MaximumBarsLookBack.Infinite;

				ResetAtStartOfDay			= true;
				BullishColor				= Brushes.Cyan;
				BearishColor				= Brushes.Magenta;
				BarThickness				= 2;

				AddPlot(new Stroke(Brushes.Transparent, 0), PlotStyle.Line, "CVD");
			}
			else if (State == State.Configure)
			{
				AddDataSeries(BarsPeriodType.Tick, 1);
			}
			else if (State == State.DataLoaded)
			{
				currentAskVolume	= 0;
				currentBidVolume	= 0;
				currentDelta		= 0;
				previousClose		= 0;
				lastTradePrice		= 0;
				currentAsk			= 0;
				currentBid			= 0;
				lastBarIndex		= -1;
				deltaHigh			= double.MinValue;
				deltaLow			= double.MaxValue;

				cvdOpen		= new Series<double>(this);
				cvdHigh		= new Series<double>(this);
				cvdLow		= new Series<double>(this);
				cvdClose	= new Series<double>(this);

				sessionIterator		= new SessionIterator(Bars);
				currentSessionBegin	= DateTime.MinValue;
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

					currentAskVolume	= 0;
					currentBidVolume	= 0;
					currentDelta		= 0;
					deltaHigh			= double.MinValue;
					deltaLow			= double.MaxValue;

					lastBarIndex = primaryBar;
				}

				double tradePrice	= Closes[1][0];
				double tradeVolume	= Volumes[1][0];

				// Classify: bid/ask when available, uptick/downtick as fallback
				if (currentAsk > 0 && currentBid > 0)
				{
					// We have valid bid/ask data
					if (tradePrice >= currentAsk)
						currentAskVolume += tradeVolume;
					else if (tradePrice <= currentBid)
						currentBidVolume += tradeVolume;
					else
					{
						// Trade between bid and ask - use uptick/downtick
						if (lastTradePrice > 0 && tradePrice > lastTradePrice)
							currentAskVolume += tradeVolume;
						else if (lastTradePrice > 0 && tradePrice < lastTradePrice)
							currentBidVolume += tradeVolume;
						// If price unchanged, don't assign (neutral)
					}
				}
				else
				{
					// No bid/ask data - fall back to pure uptick/downtick
					if (lastTradePrice > 0)
					{
						if (tradePrice > lastTradePrice)
							currentAskVolume += tradeVolume;
						else if (tradePrice < lastTradePrice)
							currentBidVolume += tradeVolume;
						// If price unchanged, don't assign (neutral)
					}
					// First trade with no reference - don't assign
				}

				lastTradePrice = tradePrice;
				currentDelta = currentAskVolume - currentBidVolume;

				// Track running high/low of cumulative delta during bar formation
				double runningCumulativeDelta = previousClose + currentDelta;
				deltaHigh = Math.Max(deltaHigh, runningCumulativeDelta);
				deltaLow = Math.Min(deltaLow, runningCumulativeDelta);
			}
			else if (BarsInProgress == 0)
			{
				if (CurrentBar < 0)
					return;

				// Detect new session using SessionIterator
				if (Bars.IsFirstBarOfSession)
				{
					sessionIterator.GetNextSession(Time[0], true);
					DateTime sessionBegin = sessionIterator.ActualSessionBegin;

					if (sessionBegin != currentSessionBegin)
					{
						if (ResetAtStartOfDay && currentSessionBegin != DateTime.MinValue)
							previousClose = 0;

						currentSessionBegin = sessionBegin;
					}
				}

				if (CurrentBar == 0)
					previousClose = 0;

				double close = previousClose + currentDelta;
				double high  = deltaHigh == double.MinValue ? close : deltaHigh;
				double low   = deltaLow == double.MaxValue ? close : deltaLow;
				double open  = Math.Min(Math.Max(previousClose, low), high);

				cvdOpen[0]	= open;
				cvdHigh[0]	= high;
				cvdLow[0]	= low;
				cvdClose[0]	= close;
				Value[0]	= close;
			}
		}

		public override void OnCalculateMinMax()
		{
			MinValue = double.MaxValue;
			MaxValue = double.MinValue;

			for (int index = ChartBars.FromIndex; index <= ChartBars.ToIndex; index++)
			{
				double high = cvdHigh.GetValueAt(index);
				double low  = cvdLow.GetValueAt(index);

				MaxValue = Math.Max(MaxValue, high);
				MinValue = Math.Min(MinValue, low);
			}
		}

		protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
		{
			if (ChartBars == null || cvdOpen == null)
				return;

			float barWidth = (float)chartControl.BarWidth;

			for (int index = ChartBars.FromIndex; index <= ChartBars.ToIndex; index++)
			{
				double open  = cvdOpen.GetValueAt(index);
				double high  = cvdHigh.GetValueAt(index);
				double low   = cvdLow.GetValueAt(index);
				double close = cvdClose.GetValueAt(index);

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
		[Display(Name = "Reset At Start Of Day", Description = "Reset cumulative delta at the start of each trading day", Order = 1, GroupName = "Parameters")]
		public bool ResetAtStartOfDay { get; set; }

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
		[Range(1, 10)]
		[Display(Name = "Bar Thickness", Description = "Thickness of the OHLC bar lines", Order = 4, GroupName = "Parameters")]
		public int BarThickness { get; set; }

		#endregion
	}
}
