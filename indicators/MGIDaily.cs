using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.DrawingTools;

namespace NinjaTrader.NinjaScript.Indicators.WolfToolz
{
	public class MGIDaily : Indicator
	{
		#region Private Fields

		// Session iterators
		private SessionIterator sessionIterator;
		private SessionIterator rthSessionIterator;
		private DateTime currentTradingDay;
		private DateTime currentRthSessionBegin;

		// Session state
		private int rthOpenBarIndex;
		private int ethOpenBarIndex;
		private bool isInRth;
		private bool isInOrPeriod;
		private bool isInIbPeriod;
		private bool orComplete;
		private bool ibComplete;
		private bool rthDetected;

		// Current Day RTH
		private double rthOpen;
		private double rthHigh;
		private double rthLow;

		// Current Day ETH
		private double ethOpen;
		private double ethHigh;
		private double ethLow;

		// Prior Day RTH
		private double priorRthOpen;
		private double priorRthHigh;
		private double priorRthLow;
		private double priorRthClose;

		// Prior Day ETH
		private double priorEthOpen;
		private double priorEthHigh;
		private double priorEthLow;
		private double priorEthClose;

		// Overnight Range
		private double onHigh;
		private double onLow;

		// Opening Range
		private double orHigh;
		private double orLow;

		// Initial Balance
		private double ibHigh;
		private double ibLow;

		// IB Extensions (computed once IB is complete)
		private double ibRange;
		private double ibExtHighX1, ibExtHighX1_5, ibExtHighX2;
		private double ibExtLowX1, ibExtLowX1_5, ibExtLowX2;

		// VWAP
		private double cumTypicalPriceVolume;
		private double cumVolume;

		// Half Gap
		private double halfGap;

		// Track whether prior day data is available
		private bool hasPriorRth;
		private bool hasPriorEth;

		#endregion

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description					= "Market Generated Information Daily Levels";
				Name						= "MGIDaily";
				Calculate					= Calculate.OnBarClose;
				IsOverlay					= true;
				DrawOnPricePanel			= true;
				DisplayInDataBox			= false;
				IsSuspendedWhileInactive	= true;
				MaximumBarsLookBack			= MaximumBarsLookBack.Infinite;

				// Parameters
				ORPeriodMinutes				= 30;
				RthSessionTemplate			= "CME US Index Futures RTH";
				ShowLabels					= true;
				LabelOffset					= 0;

				// Show/Hide
				ShowOvernightRange			= true;
				ShowOpeningRange			= true;
				ShowInitialBalance			= true;
				ShowIBExtensions			= true;
				ShowCurrentRTH				= true;
				ShowCurrentETH				= false;
				ShowPriorDayRTH				= true;
				ShowPriorDayETH				= false;
				ShowVWAP					= true;
				ShowHalfGap					= true;

				// Colors
				ONColor						= Brushes.DodgerBlue;
				ORColor						= Brushes.Orange;
				IBColor						= Brushes.MediumPurple;
				IBExtColor					= new SolidColorBrush(System.Windows.Media.Color.FromArgb(128, 147, 112, 219));
				IBExtColor.Freeze();
				CurrentRTHColor				= Brushes.White;
				CurrentETHColor				= Brushes.Gray;
				PriorRTHColor				= Brushes.Yellow;
				PriorETHColor				= Brushes.DarkGoldenrod;
				VWAPColor					= Brushes.Magenta;
				HalfGapColor				= Brushes.LimeGreen;

				AddPlot(new Stroke(Brushes.Magenta, 2), PlotStyle.Line, "VWAP");
			}
			else if (State == State.Configure)
			{
				AddDataSeries(Instrument.FullName, new BarsPeriod { BarsPeriodType = BarsPeriodType.Minute, Value = 1440 }, RthSessionTemplate);
			}
			else if (State == State.DataLoaded)
			{
				sessionIterator			= new SessionIterator(Bars);
				rthSessionIterator		= new SessionIterator(BarsArray[1]);
				currentTradingDay		= DateTime.MinValue;
				currentRthSessionBegin	= DateTime.MinValue;

				ResetEthTracking(0);
				ResetRthTracking(0);
				ResetOvernightTracking();

				hasPriorRth		= false;
				hasPriorEth		= false;
				isInRth			= false;
				rthDetected		= false;
				orComplete		= false;
				ibComplete		= false;
			}
		}

		protected override void OnBarUpdate()
		{
			if (BarsInProgress != 0)
				return;

			if (CurrentBar < 1)
				return;

			// --- Session Detection ---
			if (Bars.IsFirstBarOfSession)
			{
				sessionIterator.GetNextSession(Time[0], true);
				DateTime tradingDay = sessionIterator.ActualTradingDayExchange;

				// New ETH session (new trading day)
				if (tradingDay != currentTradingDay)
				{
					// Save current ETH as prior (if we had data)
					if (currentTradingDay != DateTime.MinValue && ethHigh != double.MinValue)
					{
						priorEthOpen	= ethOpen;
						priorEthHigh	= ethHigh;
						priorEthLow		= ethLow;
						priorEthClose	= Closes[0][1]; // prior bar's close
						hasPriorEth		= true;
					}

					currentTradingDay = tradingDay;
					rthDetected = false;
					isInRth = false;

					// Reset ETH tracking
					ResetEthTracking(Open[0]);
					ethOpenBarIndex = CurrentBar;

					// Reset overnight tracking
					ResetOvernightTracking();
				}

				// Update RTH session begin time
				rthSessionIterator.GetNextSession(Time[0], true);
				currentRthSessionBegin = rthSessionIterator.ActualSessionBegin;
			}

			// Detect RTH open
			if (!rthDetected && currentRthSessionBegin != DateTime.MinValue && Time[0] >= currentRthSessionBegin)
			{
				// Save current RTH as prior (if we had data)
				if (rthHigh != double.MinValue)
				{
					priorRthOpen	= rthOpen;
					priorRthHigh	= rthHigh;
					priorRthLow		= rthLow;
					priorRthClose	= CurrentBar > 0 ? Closes[0][1] : Close[0];
					hasPriorRth		= true;
				}

				// Capture overnight range: ETH data from ETH open to RTH open
				// ON high/low were tracked during the pre-RTH ETH portion
				// (already accumulated in onHigh/onLow during ETH bars before RTH)

				// Reset RTH tracking
				ResetRthTracking(Open[0]);
				rthOpenBarIndex = CurrentBar;
				rthDetected = true;
				isInRth = true;

				// Compute half gap
				if (hasPriorRth)
					halfGap = (priorRthClose + rthOpen) / 2.0;

				// Start OR and IB periods
				isInOrPeriod = true;
				isInIbPeriod = true;
				orComplete = false;
				ibComplete = false;
				orHigh = High[0];
				orLow = Low[0];
				ibHigh = High[0];
				ibLow = Low[0];

				// Reset VWAP
				cumTypicalPriceVolume = 0;
				cumVolume = 0;
			}

			// --- Update ETH High/Low ---
			if (ethHigh == double.MinValue)
			{
				ethHigh = High[0];
				ethLow = Low[0];
			}
			else
			{
				ethHigh = Math.Max(ethHigh, High[0]);
				ethLow = Math.Min(ethLow, Low[0]);
			}

			// --- Update Overnight Range (pre-RTH bars within ETH) ---
			if (!rthDetected)
			{
				if (onHigh == double.MinValue)
				{
					onHigh = High[0];
					onLow = Low[0];
				}
				else
				{
					onHigh = Math.Max(onHigh, High[0]);
					onLow = Math.Min(onLow, Low[0]);
				}
			}

			// --- RTH Tracking ---
			if (isInRth && rthDetected)
			{
				rthHigh = Math.Max(rthHigh, High[0]);
				rthLow = Math.Min(rthLow, Low[0]);

				// Time elapsed since RTH open
				double minutesSinceRthOpen = (Time[0] - currentRthSessionBegin).TotalMinutes;

				// Opening Range period
				if (isInOrPeriod && !orComplete)
				{
					orHigh = Math.Max(orHigh, High[0]);
					orLow = Math.Min(orLow, Low[0]);

					if (minutesSinceRthOpen >= ORPeriodMinutes)
					{
						orComplete = true;
						isInOrPeriod = false;
					}
				}

				// Initial Balance period (first 60 minutes)
				if (isInIbPeriod && !ibComplete)
				{
					ibHigh = Math.Max(ibHigh, High[0]);
					ibLow = Math.Min(ibLow, Low[0]);

					if (minutesSinceRthOpen >= 60)
					{
						ibComplete = true;
						isInIbPeriod = false;

						// Compute IB extensions
						ibRange = ibHigh - ibLow;
						ibExtHighX1		= ibHigh + 1.0 * ibRange;
						ibExtHighX1_5	= ibHigh + 1.5 * ibRange;
						ibExtHighX2		= ibHigh + 2.0 * ibRange;
						ibExtLowX1		= ibLow - 1.0 * ibRange;
						ibExtLowX1_5	= ibLow - 1.5 * ibRange;
						ibExtLowX2		= ibLow - 2.0 * ibRange;
					}
				}

				// VWAP calculation
				double typicalPrice = (High[0] + Low[0] + Close[0]) / 3.0;
				double vol = Volume[0];
				cumTypicalPriceVolume += typicalPrice * vol;
				cumVolume += vol;
			}

			// --- Set VWAP plot ---
			if (ShowVWAP && isInRth && rthDetected && cumVolume > 0)
			{
				Values[0][0] = cumTypicalPriceVolume / cumVolume;
				PlotBrushes[0][0] = VWAPColor;
			}
			else
			{
				Values[0][0] = Close[0]; // park at close to avoid scale distortion
				PlotBrushes[0][0] = Brushes.Transparent;
			}

			// --- Draw Levels ---
			DrawAllLevels();

			// --- Insufficient data warning ---
			bool needsPrior = ShowPriorDayRTH || ShowPriorDayETH || ShowHalfGap;
			bool needsRth   = ShowOpeningRange || ShowInitialBalance || ShowIBExtensions || ShowCurrentRTH || ShowVWAP;

			if ((needsPrior && !hasPriorRth) || (needsRth && !rthDetected))
			{
				Draw.TextFixed(this, "MGI_Warning",
					"MGI Daily: Load more data (at least 2 full trading days) for all levels to appear.",
					TextPosition.BottomRight, Brushes.Orange, new Gui.Tools.SimpleFont("Arial", 10),
					Brushes.Transparent, Brushes.Transparent, 0);
			}
			else
			{
				RemoveDrawObject("MGI_Warning");
			}
		}

		#region Drawing

		private void DrawAllLevels()
		{
			int startBar;
			int endBar = CurrentBar;

			// --- Overnight Range ---
			if (ShowOvernightRange && onHigh != double.MinValue)
			{
				startBar = ethOpenBarIndex;
				double onMid = (onHigh + onLow) / 2.0;
				DrawLevel("MGI_ON_High", "ONH", onHigh, startBar, endBar, ONColor, DashStyleHelper.Solid);
				DrawLevel("MGI_ON_Low", "ONL", onLow, startBar, endBar, ONColor, DashStyleHelper.Solid);
				DrawLevel("MGI_ON_Mid", "ONMid", onMid, startBar, endBar, ONColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_ON_High"); RemoveLevel("MGI_ON_Low"); RemoveLevel("MGI_ON_Mid");
				RemoveLabel("MGI_ON_High"); RemoveLabel("MGI_ON_Low"); RemoveLabel("MGI_ON_Mid");
			}

			// --- Opening Range ---
			if (ShowOpeningRange && rthDetected && (orComplete || isInOrPeriod))
			{
				startBar = rthOpenBarIndex;
				double orMid = (orHigh + orLow) / 2.0;
				DrawLevel("MGI_OR_High", "ORH", orHigh, startBar, endBar, ORColor, DashStyleHelper.Solid);
				DrawLevel("MGI_OR_Low", "ORL", orLow, startBar, endBar, ORColor, DashStyleHelper.Solid);
				DrawLevel("MGI_OR_Mid", "ORMid", orMid, startBar, endBar, ORColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_OR_High"); RemoveLevel("MGI_OR_Low"); RemoveLevel("MGI_OR_Mid");
				RemoveLabel("MGI_OR_High"); RemoveLabel("MGI_OR_Low"); RemoveLabel("MGI_OR_Mid");
			}

			// --- Initial Balance ---
			if (ShowInitialBalance && rthDetected && (ibComplete || isInIbPeriod))
			{
				startBar = rthOpenBarIndex;
				double ibMid = (ibHigh + ibLow) / 2.0;
				DrawLevel("MGI_IB_High", "IBH", ibHigh, startBar, endBar, IBColor, DashStyleHelper.Solid);
				DrawLevel("MGI_IB_Low", "IBL", ibLow, startBar, endBar, IBColor, DashStyleHelper.Solid);
				DrawLevel("MGI_IB_Mid", "IBMid", ibMid, startBar, endBar, IBColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_IB_High"); RemoveLevel("MGI_IB_Low"); RemoveLevel("MGI_IB_Mid");
				RemoveLabel("MGI_IB_High"); RemoveLabel("MGI_IB_Low"); RemoveLabel("MGI_IB_Mid");
			}

			// --- IB Extensions ---
			if (ShowIBExtensions && ibComplete)
			{
				startBar = rthOpenBarIndex;
				DrawLevel("MGI_IBExt_H1",   "IB+1x",   ibExtHighX1,   startBar, endBar, IBExtColor, DashStyleHelper.Dash);
				DrawLevel("MGI_IBExt_H1_5", "IB+1.5x", ibExtHighX1_5, startBar, endBar, IBExtColor, DashStyleHelper.Dash);
				DrawLevel("MGI_IBExt_H2",   "IB+2x",   ibExtHighX2,   startBar, endBar, IBExtColor, DashStyleHelper.Dash);
				DrawLevel("MGI_IBExt_L1",   "IB-1x",   ibExtLowX1,    startBar, endBar, IBExtColor, DashStyleHelper.Dash);
				DrawLevel("MGI_IBExt_L1_5", "IB-1.5x", ibExtLowX1_5,  startBar, endBar, IBExtColor, DashStyleHelper.Dash);
				DrawLevel("MGI_IBExt_L2",   "IB-2x",   ibExtLowX2,    startBar, endBar, IBExtColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_IBExt_H1"); RemoveLevel("MGI_IBExt_H1_5"); RemoveLevel("MGI_IBExt_H2");
				RemoveLevel("MGI_IBExt_L1"); RemoveLevel("MGI_IBExt_L1_5"); RemoveLevel("MGI_IBExt_L2");
				RemoveLabel("MGI_IBExt_H1"); RemoveLabel("MGI_IBExt_H1_5"); RemoveLabel("MGI_IBExt_H2");
				RemoveLabel("MGI_IBExt_L1"); RemoveLabel("MGI_IBExt_L1_5"); RemoveLabel("MGI_IBExt_L2");
			}

			// --- Current Day RTH ---
			if (ShowCurrentRTH && rthDetected && rthHigh != double.MinValue)
			{
				startBar = rthOpenBarIndex;
				double rthMid = (rthHigh + rthLow) / 2.0;
				DrawLevel("MGI_CRTH_Open", "RTH Open", rthOpen, startBar, endBar, CurrentRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_CRTH_High", "RTH Hi",   rthHigh, startBar, endBar, CurrentRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_CRTH_Low",  "RTH Lo",   rthLow,  startBar, endBar, CurrentRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_CRTH_Mid",  "RTH Mid",  rthMid,  startBar, endBar, CurrentRTHColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_CRTH_Open"); RemoveLevel("MGI_CRTH_High"); RemoveLevel("MGI_CRTH_Low"); RemoveLevel("MGI_CRTH_Mid");
				RemoveLabel("MGI_CRTH_Open"); RemoveLabel("MGI_CRTH_High"); RemoveLabel("MGI_CRTH_Low"); RemoveLabel("MGI_CRTH_Mid");
			}

			// --- Current Day ETH ---
			if (ShowCurrentETH && ethHigh != double.MinValue)
			{
				startBar = ethOpenBarIndex;
				double ethMid = (ethHigh + ethLow) / 2.0;
				DrawLevel("MGI_CETH_Open", "ETH Open", ethOpen, startBar, endBar, CurrentETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_CETH_High", "ETH Hi",   ethHigh, startBar, endBar, CurrentETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_CETH_Low",  "ETH Lo",   ethLow,  startBar, endBar, CurrentETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_CETH_Mid",  "ETH Mid",  ethMid,  startBar, endBar, CurrentETHColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_CETH_Open"); RemoveLevel("MGI_CETH_High"); RemoveLevel("MGI_CETH_Low"); RemoveLevel("MGI_CETH_Mid");
				RemoveLabel("MGI_CETH_Open"); RemoveLabel("MGI_CETH_High"); RemoveLabel("MGI_CETH_Low"); RemoveLabel("MGI_CETH_Mid");
			}

			// --- Prior Day RTH ---
			if (ShowPriorDayRTH && hasPriorRth)
			{
				startBar = ethOpenBarIndex;
				double priorRthMid = (priorRthHigh + priorRthLow) / 2.0;
				DrawLevel("MGI_PRTH_Open",  "pRTH Open",  priorRthOpen,  startBar, endBar, PriorRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PRTH_High",  "pRTH Hi",    priorRthHigh,  startBar, endBar, PriorRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PRTH_Low",   "pRTH Lo",    priorRthLow,   startBar, endBar, PriorRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PRTH_Close", "pRTH Close", priorRthClose, startBar, endBar, PriorRTHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PRTH_Mid",   "pRTH Mid",   priorRthMid,   startBar, endBar, PriorRTHColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_PRTH_Open"); RemoveLevel("MGI_PRTH_High"); RemoveLevel("MGI_PRTH_Low");
				RemoveLevel("MGI_PRTH_Close"); RemoveLevel("MGI_PRTH_Mid");
				RemoveLabel("MGI_PRTH_Open"); RemoveLabel("MGI_PRTH_High"); RemoveLabel("MGI_PRTH_Low");
				RemoveLabel("MGI_PRTH_Close"); RemoveLabel("MGI_PRTH_Mid");
			}

			// --- Prior Day ETH ---
			if (ShowPriorDayETH && hasPriorEth)
			{
				startBar = ethOpenBarIndex;
				double priorEthMid = (priorEthHigh + priorEthLow) / 2.0;
				DrawLevel("MGI_PETH_Open",  "pETH Open",  priorEthOpen,  startBar, endBar, PriorETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PETH_High",  "pETH Hi",    priorEthHigh,  startBar, endBar, PriorETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PETH_Low",   "pETH Lo",    priorEthLow,   startBar, endBar, PriorETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PETH_Close", "pETH Close", priorEthClose, startBar, endBar, PriorETHColor, DashStyleHelper.Solid);
				DrawLevel("MGI_PETH_Mid",   "pETH Mid",   priorEthMid,   startBar, endBar, PriorETHColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_PETH_Open"); RemoveLevel("MGI_PETH_High"); RemoveLevel("MGI_PETH_Low");
				RemoveLevel("MGI_PETH_Close"); RemoveLevel("MGI_PETH_Mid");
				RemoveLabel("MGI_PETH_Open"); RemoveLabel("MGI_PETH_High"); RemoveLabel("MGI_PETH_Low");
				RemoveLabel("MGI_PETH_Close"); RemoveLabel("MGI_PETH_Mid");
			}

			// --- Half Gap ---
			if (ShowHalfGap && hasPriorRth && rthDetected)
			{
				startBar = rthOpenBarIndex;
				DrawLevel("MGI_HalfGap", "HalfGap", halfGap, startBar, endBar, HalfGapColor, DashStyleHelper.Dash);
			}
			else
			{
				RemoveLevel("MGI_HalfGap");
				RemoveLabel("MGI_HalfGap");
			}
		}

		private void DrawLevel(string tag, string label, double price, int startBar, int endBar, Brush color, DashStyleHelper dashStyle)
		{
			Draw.Line(this, tag, false, CurrentBar - startBar, price, CurrentBar - endBar, price, color, dashStyle, 1);

			if (ShowLabels)
			{
				string labelTag = "MGI_LBL_" + tag.Substring(4); // strip "MGI_" prefix and re-add via LBL
				Draw.Text(this, labelTag, false, label, CurrentBar - endBar + LabelOffset, price, 0, color,
					new Gui.Tools.SimpleFont("Arial", 10), TextAlignment.Left, Brushes.Transparent, Brushes.Transparent, 0);
			}
		}

		private void RemoveLevel(string tag)
		{
			RemoveDrawObject(tag);
		}

		private void RemoveLabel(string tag)
		{
			string labelTag = "MGI_LBL_" + tag.Substring(4);
			RemoveDrawObject(labelTag);
		}

		#endregion

		#region Helpers

		private void ResetEthTracking(double openPrice)
		{
			ethOpen		= openPrice;
			ethHigh		= double.MinValue;
			ethLow		= double.MaxValue;
		}

		private void ResetRthTracking(double openPrice)
		{
			rthOpen		= openPrice;
			rthHigh		= double.MinValue;
			rthLow		= double.MaxValue;

			orHigh		= double.MinValue;
			orLow		= double.MaxValue;
			ibHigh		= double.MinValue;
			ibLow		= double.MaxValue;

			ibRange		= 0;
			ibExtHighX1 = 0; ibExtHighX1_5 = 0; ibExtHighX2 = 0;
			ibExtLowX1  = 0; ibExtLowX1_5  = 0; ibExtLowX2  = 0;

			cumTypicalPriceVolume	= 0;
			cumVolume				= 0;
			halfGap					= 0;
		}

		private void ResetOvernightTracking()
		{
			onHigh	= double.MinValue;
			onLow	= double.MaxValue;
		}

		#endregion

		#region Properties

		// --- Parameters ---

		[NinjaScriptProperty]
		[Range(1, 120)]
		[Display(Name = "OR Period (Minutes)", Description = "Opening Range period in minutes from RTH open", Order = 1, GroupName = "Parameters")]
		public int ORPeriodMinutes { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "RTH Session Template", Description = "RTH session template name for session detection", Order = 2, GroupName = "Parameters")]
		public string RthSessionTemplate { get; set; }

		[Display(Name = "Show Labels", Description = "Show text labels on lines", Order = 3, GroupName = "Parameters")]
		public bool ShowLabels { get; set; }

		[Range(-10, 10)]
		[Display(Name = "Label Offset", Description = "Bars offset for label placement from line start", Order = 4, GroupName = "Parameters")]
		public int LabelOffset { get; set; }

		// --- Show/Hide ---

		[Display(Name = "Overnight Range", Description = "Show ONH, ONL, ONMid", Order = 1, GroupName = "Show/Hide")]
		public bool ShowOvernightRange { get; set; }

		[Display(Name = "Opening Range", Description = "Show ORH, ORL, ORMid", Order = 2, GroupName = "Show/Hide")]
		public bool ShowOpeningRange { get; set; }

		[Display(Name = "Initial Balance", Description = "Show IBH, IBL, IBMid", Order = 3, GroupName = "Show/Hide")]
		public bool ShowInitialBalance { get; set; }

		[Display(Name = "IB Extensions", Description = "Show IB extension levels", Order = 4, GroupName = "Show/Hide")]
		public bool ShowIBExtensions { get; set; }

		[Display(Name = "Current RTH", Description = "Show current day RTH Open/High/Low/Mid", Order = 5, GroupName = "Show/Hide")]
		public bool ShowCurrentRTH { get; set; }

		[Display(Name = "Current ETH", Description = "Show current day ETH Open/High/Low/Mid", Order = 6, GroupName = "Show/Hide")]
		public bool ShowCurrentETH { get; set; }

		[Display(Name = "Prior Day RTH", Description = "Show prior day RTH OHLC/Mid", Order = 7, GroupName = "Show/Hide")]
		public bool ShowPriorDayRTH { get; set; }

		[Display(Name = "Prior Day ETH", Description = "Show prior day ETH OHLC/Mid", Order = 8, GroupName = "Show/Hide")]
		public bool ShowPriorDayETH { get; set; }

		[Display(Name = "VWAP", Description = "Show RTH VWAP line", Order = 9, GroupName = "Show/Hide")]
		public bool ShowVWAP { get; set; }

		[Display(Name = "Half Gap", Description = "Show half gap level", Order = 10, GroupName = "Show/Hide")]
		public bool ShowHalfGap { get; set; }

		// --- Colors ---

		[XmlIgnore]
		[Display(Name = "Overnight", Order = 1, GroupName = "Colors")]
		public Brush ONColor { get; set; }
		[Browsable(false)]
		public string ONColorSerialize
		{
			get { return Serialize.BrushToString(ONColor); }
			set { ONColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Opening Range", Order = 2, GroupName = "Colors")]
		public Brush ORColor { get; set; }
		[Browsable(false)]
		public string ORColorSerialize
		{
			get { return Serialize.BrushToString(ORColor); }
			set { ORColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Initial Balance", Order = 3, GroupName = "Colors")]
		public Brush IBColor { get; set; }
		[Browsable(false)]
		public string IBColorSerialize
		{
			get { return Serialize.BrushToString(IBColor); }
			set { IBColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "IB Extensions", Order = 4, GroupName = "Colors")]
		public Brush IBExtColor { get; set; }
		[Browsable(false)]
		public string IBExtColorSerialize
		{
			get { return Serialize.BrushToString(IBExtColor); }
			set { IBExtColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Current RTH", Order = 5, GroupName = "Colors")]
		public Brush CurrentRTHColor { get; set; }
		[Browsable(false)]
		public string CurrentRTHColorSerialize
		{
			get { return Serialize.BrushToString(CurrentRTHColor); }
			set { CurrentRTHColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Current ETH", Order = 6, GroupName = "Colors")]
		public Brush CurrentETHColor { get; set; }
		[Browsable(false)]
		public string CurrentETHColorSerialize
		{
			get { return Serialize.BrushToString(CurrentETHColor); }
			set { CurrentETHColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Prior RTH", Order = 7, GroupName = "Colors")]
		public Brush PriorRTHColor { get; set; }
		[Browsable(false)]
		public string PriorRTHColorSerialize
		{
			get { return Serialize.BrushToString(PriorRTHColor); }
			set { PriorRTHColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Prior ETH", Order = 8, GroupName = "Colors")]
		public Brush PriorETHColor { get; set; }
		[Browsable(false)]
		public string PriorETHColorSerialize
		{
			get { return Serialize.BrushToString(PriorETHColor); }
			set { PriorETHColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "VWAP", Order = 9, GroupName = "Colors")]
		public Brush VWAPColor { get; set; }
		[Browsable(false)]
		public string VWAPColorSerialize
		{
			get { return Serialize.BrushToString(VWAPColor); }
			set { VWAPColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Half Gap", Order = 10, GroupName = "Colors")]
		public Brush HalfGapColor { get; set; }
		[Browsable(false)]
		public string HalfGapColorSerialize
		{
			get { return Serialize.BrushToString(HalfGapColor); }
			set { HalfGapColor = Serialize.StringToBrush(value); }
		}

		#endregion
	}
}
