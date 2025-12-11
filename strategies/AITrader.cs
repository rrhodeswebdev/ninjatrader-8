using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;

namespace NinjaTrader.NinjaScript.Strategies
{
    public class AITrader : Strategy
    {
        private HttpClient httpClient;
        private string serverUrl = "https://118cc023e21f.ngrok-free.app/analysis"; // Change to "http://YOUR_LOCAL_IP:8000/analysis" for VPS access
        private bool historicalDataSent = false;
        private int historicalBarsToSend = 2000; // Configurable parameter (1 week of 1-min bars)

        // Trading direction flags
        private bool allowLongTrades = true;
        private bool allowShortTrades = true;
        private bool tradingEnabled = true;
        private readonly TimeSpan tradingStartTime = new TimeSpan(9, 30, 0);  // Gate live trading until 9:30

        // Daily Profit/Loss Management
        private double dailyProfitGoal = 500.0;    // Daily profit goal in currency
        private double dailyMaxLoss = 250.0;       // Daily max loss in currency
        private double dailyPnL = 0.0;             // Current day's P&L
        private DateTime currentDay = DateTime.MinValue;  // Track current trading day

        // Position tracking to prevent duplicate entries
        private bool isLongOrderPending = false;
        private bool isShortOrderPending = false;

        // Track desired position for reversals
        private string desiredPosition = "flat";  // "flat", "long", or "short"
        private double desiredStopLoss = 0.0;  // SL for delayed entry
        private double desiredTakeProfit = 0.0;  // TP for delayed entry



        // Dashboard state tracking
        private string lastSignal = "N/A";
        private double lastConfidence = 0.0;
        private DateTime lastSignalTime = DateTime.MinValue;
        private int totalSignals = 0;
        private int longSignals = 0;
        private int shortSignals = 0;
        private int holdSignals = 0;

        // Server connection tracking
        private bool serverConnected = false;
        private string serverStatus = "Connecting...";

        // Dashboard rendering resources
        private System.Windows.Media.Typeface dashboardTypeface;
        private System.Windows.Media.SolidColorBrush backgroundBrush;
        private System.Windows.Media.SolidColorBrush textBrush;
        private System.Windows.Media.SolidColorBrush headerBrush;
        private System.Windows.Media.SolidColorBrush longBrush;
        private System.Windows.Media.SolidColorBrush shortBrush;
        private System.Windows.Media.SolidColorBrush holdBrush;

        // Cached SharpDX objects for performance
        private SharpDX.DirectWrite.TextFormat cachedTextFormat;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "AI Trading Strategy with External Server Communication";
                Name = "AITrader";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.UniqueEntries;
                IsExitOnSessionCloseStrategy = true;
                ExitOnSessionCloseSeconds = 30;
                IsFillLimitOnTouch = false;
                MaximumBarsLookBack = MaximumBarsLookBack.Infinite;
                OrderFillResolution = OrderFillResolution.Standard;
                Slippage = 0;
                StartBehavior = StartBehavior.WaitUntilFlat;
                TimeInForce = TimeInForce.Gtc;
                TraceOrders = false;
                RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade = 20;
                IsInstantiatedOnEachOptimizationIteration = true;

                // Default historical bars to send (can be changed in strategy parameters)
                // 25000 bars = ~17 trading days of 1-min ES data (better for trend learning)
                // IMPROVED: Increased from 15000 to 25000 for better trend pattern recognition
                HistoricalBarsToSend = 25000;

                // Default Daily Profit/Loss Goals
                DailyProfitGoal = 500.0;  // Default $500 daily profit goal
                DailyMaxLoss = 250.0;     // Default $250 daily max loss
            }
            else if (State == State.Configure)
            {
                // Add secondary 5-minute bars for multi-timeframe analysis
                // BarsArray[0] = primary timeframe (e.g., 1-min)
                // BarsArray[1] = secondary timeframe (5-min)
                AddDataSeries(BarsPeriodType.Minute, 5);

                // Initialize HTTP client
                httpClient = new HttpClient();
                httpClient.Timeout = TimeSpan.FromSeconds(30);
            }
            else if (State == State.DataLoaded)
            {
                // Initialize dashboard brushes and fonts
                dashboardTypeface = new System.Windows.Media.Typeface(new System.Windows.Media.FontFamily("Arial"),
                    System.Windows.FontStyles.Normal, System.Windows.FontWeights.Bold, System.Windows.FontStretches.Normal);

                backgroundBrush = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromArgb(200, 20, 20, 30));
                textBrush = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Colors.White);
                headerBrush = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Color.FromArgb(255, 100, 150, 255));
                longBrush = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Colors.LimeGreen);
                shortBrush = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Colors.Red);
                holdBrush = new System.Windows.Media.SolidColorBrush(System.Windows.Media.Colors.Yellow);

                backgroundBrush.Freeze();
                textBrush.Freeze();
                headerBrush.Freeze();
                longBrush.Freeze();
                shortBrush.Freeze();
                holdBrush.Freeze();

                // Cache SharpDX TextFormat for reuse (major performance improvement)
                cachedTextFormat = new SharpDX.DirectWrite.TextFormat(
                    Core.Globals.DirectWriteFactory, "Arial", SharpDX.DirectWrite.FontWeight.Bold,
                    SharpDX.DirectWrite.FontStyle.Normal, 16);
            }
            else if (State == State.Terminated)
            {
                // Clean up resources
                if (httpClient != null)
                {
                    httpClient.Dispose();
                    httpClient = null;
                }

                // Clean up cached SharpDX objects
                if (cachedTextFormat != null)
                {
                    cachedTextFormat.Dispose();
                    cachedTextFormat = null;
                }
            }
        }

        protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
        {
            base.OnRender(chartControl, chartScale);

            // Skip rendering if cached format not initialized yet
            if (cachedTextFormat == null)
                return;

            // Dashboard position (top-left corner)
            double x = 20;
            double y = 60;
            double width = 400;
            double lineHeight = 28;

            // Create the background rectangle (increased height for daily P&L info)
            SharpDX.RectangleF backgroundRect = new SharpDX.RectangleF((float)x, (float)y, (float)width, (float)(lineHeight * 17));
            RenderTarget.FillRectangle(backgroundRect, backgroundBrush.ToDxBrush(RenderTarget));

            // Optimized helper function - reuses cached TextFormat, only creates/disposes TextLayout
            System.Action<string, double, double, System.Windows.Media.Brush> drawText = (text, textX, textY, brush) =>
            {
                SharpDX.DirectWrite.TextLayout textLayout = new SharpDX.DirectWrite.TextLayout(
                    Core.Globals.DirectWriteFactory, text, cachedTextFormat, (float)width, (float)lineHeight);

                RenderTarget.DrawTextLayout(new SharpDX.Vector2((float)textX, (float)textY),
                    textLayout, brush.ToDxBrush(RenderTarget), SharpDX.Direct2D1.DrawTextOptions.None);

                textLayout.Dispose();
            };

            double currentY = y + 8;

            // Header
            drawText("AI TRADER DASHBOARD", x + 15, currentY, headerBrush);
            currentY += lineHeight + 8;

            // Server Status
            var statusBrush = serverConnected ? longBrush : shortBrush;
            drawText("Server: " + serverStatus, x + 15, currentY, statusBrush);
            currentY += lineHeight;

            // Trading Mode
            string mode = !tradingEnabled ? "STOPPED" :
                         (allowLongTrades && allowShortTrades) ? "BOTH" :
                         allowLongTrades ? "LONGS ONLY" : "SHORTS ONLY";
            var modeBrush = !tradingEnabled ? shortBrush :
                           (allowLongTrades && allowShortTrades) ? longBrush :
                           allowLongTrades ? longBrush : shortBrush;
            drawText("Mode: " + mode, x + 15, currentY, modeBrush);
            currentY += lineHeight + 8;

            // Last Signal
            drawText("LAST SIGNAL", x + 15, currentY, headerBrush);
            currentY += lineHeight;

            var signalBrush = lastSignal.ToUpper() == "LONG" ? longBrush :
                             lastSignal.ToUpper() == "SHORT" ? shortBrush :
                             lastSignal.ToUpper() == "HOLD" ? holdBrush : textBrush;
            drawText("Signal: " + lastSignal.ToUpper(), x + 15, currentY, signalBrush);
            currentY += lineHeight;

            drawText("Confidence: " + (lastConfidence * 100).ToString("F2") + "%", x + 15, currentY, textBrush);
            currentY += lineHeight;

            if (lastSignalTime != DateTime.MinValue)
            {
                drawText("Time: " + lastSignalTime.ToString("HH:mm:ss"), x + 15, currentY, textBrush);
            }
            currentY += lineHeight + 8;

            // Signal Statistics
            drawText("SIGNAL STATS", x + 15, currentY, headerBrush);
            currentY += lineHeight;

            drawText("Total: " + totalSignals, x + 15, currentY, textBrush);
            currentY += lineHeight;

            drawText("Long: " + longSignals + " (" + (totalSignals > 0 ? (longSignals * 100.0 / totalSignals).ToString("F1") : "0") + "%)", x + 15, currentY, longBrush);
            currentY += lineHeight;

            drawText("Short: " + shortSignals + " (" + (totalSignals > 0 ? (shortSignals * 100.0 / totalSignals).ToString("F1") : "0") + "%)", x + 15, currentY, shortBrush);
            currentY += lineHeight;

            drawText("Hold: " + holdSignals + " (" + (totalSignals > 0 ? (holdSignals * 100.0 / totalSignals).ToString("F1") : "0") + "%)", x + 15, currentY, holdBrush);
            currentY += lineHeight + 8;

            // Daily P&L Section
            drawText("DAILY P&L", x + 15, currentY, headerBrush);
            currentY += lineHeight;

            var pnlBrush = dailyPnL >= 0 ? longBrush : shortBrush;
            drawText("Current: " + dailyPnL.ToString("C2"), x + 15, currentY, pnlBrush);
            currentY += lineHeight;

            drawText("Goal: " + dailyProfitGoal.ToString("C2"), x + 15, currentY, longBrush);
            currentY += lineHeight;

            drawText("Max Loss: -" + dailyMaxLoss.ToString("C2"), x + 15, currentY, shortBrush);
            currentY += lineHeight;

            // Progress bar for daily P&L
            double progressPercent = 0.0;
            System.Windows.Media.Brush progressBrush = textBrush;

            if (dailyPnL >= 0)
            {
                progressPercent = Math.Min(100, (dailyPnL / dailyProfitGoal) * 100);
                progressBrush = longBrush;
            }
            else
            {
                progressPercent = Math.Min(100, (Math.Abs(dailyPnL) / dailyMaxLoss) * 100);
                progressBrush = shortBrush;
            }

            drawText("Progress: " + progressPercent.ToString("F1") + "%", x + 15, currentY, progressBrush);
        }

        protected override void OnOrderUpdate(Order order, double limitPrice, double stopPrice, int quantity, int filled, double averageFillPrice, OrderState orderState, DateTime time, ErrorCode error, string nativeError)
        {
            // Reset pending flags when orders are filled or rejected
            if (order.Name == "AILong")
            {
                if (orderState == OrderState.Filled || orderState == OrderState.PartFilled)
                {
                    isLongOrderPending = false;  // Order filled, clear flag
                }
                else if (orderState == OrderState.Cancelled || orderState == OrderState.Rejected)
                {
                    isLongOrderPending = false;  // Order failed, clear flag
                }
            }
            else if (order.Name == "AIShort")
            {
                if (orderState == OrderState.Filled || orderState == OrderState.PartFilled)
                {
                    isShortOrderPending = false;  // Order filled, clear flag
                }
                else if (orderState == OrderState.Cancelled || orderState == OrderState.Rejected)
                {
                    isShortOrderPending = false;  // Order failed, clear flag
                }
            }
        }

        protected override void OnExecutionUpdate(Execution execution, string executionId, double price, int quantity, MarketPosition marketPosition, string orderId, DateTime time)
        {
            // Additional safety: Clear flags on position changes
            if (execution.Order.Name == "AILong" && Position.MarketPosition == MarketPosition.Long)
            {
                isLongOrderPending = false;
                isShortOrderPending = false;
            }
            else if (execution.Order.Name == "AIShort" && Position.MarketPosition == MarketPosition.Short)
            {
                isShortOrderPending = false;
                isLongOrderPending = false;
            }

            // Reset flags when flat and update daily P&L
            if (Position.MarketPosition == MarketPosition.Flat)
            {
                isLongOrderPending = false;
                isShortOrderPending = false;

                // Update daily P&L after position is closed
                UpdateDailyPnL();

                // If we became flat and have a desired position, enter it now
                if (desiredPosition == "long" && allowLongTrades && tradingEnabled)
                {
                    Print("Position now FLAT - Entering LONG as desired");
                    Print("  Stop Loss: " + desiredStopLoss.ToString("F2") + ", Take Profit: " + desiredTakeProfit.ToString("F2"));

                    // Set SL/TP if available
                    if (desiredStopLoss > 0 && desiredTakeProfit > 0)
                    {
                        SetStopLoss("AILong", CalculationMode.Price, desiredStopLoss, false);
                        SetProfitTarget("AILong", CalculationMode.Price, desiredTakeProfit);
                    }

                    desiredPosition = "flat";
                    desiredStopLoss = 0.0;
                    desiredTakeProfit = 0.0;
                    isLongOrderPending = true;
                    EnterLong(1, "AILong");
                }
                else if (desiredPosition == "short" && allowShortTrades && tradingEnabled)
                {
                    Print("Position now FLAT - Entering SHORT as desired");
                    Print("  Stop Loss: " + desiredStopLoss.ToString("F2") + ", Take Profit: " + desiredTakeProfit.ToString("F2"));

                    // Set SL/TP if available
                    if (desiredStopLoss > 0 && desiredTakeProfit > 0)
                    {
                        SetStopLoss("AIShort", CalculationMode.Price, desiredStopLoss, false);
                        SetProfitTarget("AIShort", CalculationMode.Price, desiredTakeProfit);
                    }

                    desiredPosition = "flat";
                    desiredStopLoss = 0.0;
                    desiredTakeProfit = 0.0;
                    isShortOrderPending = true;
                    EnterShort(1, "AIShort");
                }
                else
                {
                    desiredPosition = "flat";
                    desiredStopLoss = 0.0;
                    desiredTakeProfit = 0.0;
                }
            }
        }

        private void UpdateDailyPnL()
        {
            // Reset daily P&L if it's a new trading day
            DateTime today = DateTime.Now.Date;
            if (currentDay != today)
            {
                currentDay = today;
                dailyPnL = 0.0;
                Print("=== NEW TRADING DAY - Daily P&L reset ===");
            }

            // Calculate realized P&L from closed trades today
            double realizedPnL = 0.0;
            if (SystemPerformance != null && SystemPerformance.AllTrades.Count > 0)
            {
                foreach (var trade in SystemPerformance.AllTrades)
                {
                    if (trade.Exit.Time.Date == today)
                    {
                        realizedPnL += trade.ProfitCurrency;
                    }
                }
            }

            // Add unrealized P&L from current open position
            double unrealizedPnL = Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);

            // Total daily P&L = realized + unrealized
            dailyPnL = realizedPnL + unrealizedPnL;

            Print($"[P&L] Daily: {dailyPnL:C2} | Realized: {realizedPnL:C2} | Unrealized: {unrealizedPnL:C2} | Goal: {dailyProfitGoal:C2} | Max Loss: -{dailyMaxLoss:C2}");

            // Check if we hit profit goal or max loss
            if (dailyPnL >= dailyProfitGoal)
            {
                Print($"*** DAILY PROFIT GOAL REACHED: {dailyPnL:C2} >= {dailyProfitGoal:C2} ***");
                tradingEnabled = false;

                // Exit any open positions to lock in profit
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    Print("Exiting LONG position to lock in profit");
                    ExitLong();
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    Print("Exiting SHORT position to lock in profit");
                    ExitShort();
                }
            }
            else if (dailyPnL <= -dailyMaxLoss)
            {
                Print($"*** DAILY MAX LOSS REACHED: {dailyPnL:C2} <= -{dailyMaxLoss:C2} ***");
                tradingEnabled = false;

                // Exit any open positions
                if (Position.MarketPosition == MarketPosition.Long)
                {
                    Print("Exiting LONG position due to max loss");
                    ExitLong();
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    Print("Exiting SHORT position due to max loss");
                    ExitShort();
                }
            }
        }

        protected override void OnBarUpdate()
        {
            // Only process on primary timeframe updates (BarsInProgress == 0)
            // When secondary data series is added, OnBarUpdate is called for both timeframes
            if (BarsInProgress != 0)
                return;

            // Wait until we have enough bars loaded before sending historical data
            if (!historicalDataSent)
            {
                // Check if we have enough bars OR if we're in realtime (use whatever bars are available)
                if (CurrentBar >= (historicalBarsToSend - 1) || State == State.Realtime)
                {
                    SendHistoricalData();
                    historicalDataSent = true;
                }
                else
                {
                    // Not enough bars yet, keep waiting
                    return;
                }
            }

            // Ensure we have enough bars before trading
            if (CurrentBar < BarsRequiredToTrade)
                return;

            // Update daily P&L on every bar to track profit/loss continuously
            UpdateDailyPnL();

            // Send current bar data on each bar close (real-time only)
            if (State == State.Realtime)
            {
                // Do not trade before the configured start time (uses primary series time zone)
                if (Times[0][0].TimeOfDay < tradingStartTime)
                    return;

                SendBarData(CurrentBar);
            }
        }

        private void SendHistoricalData()
        {
            // Capture bar count and data before async operation
            int barCount = Math.Min(CurrentBars[0] + 1, historicalBarsToSend);
            StringBuilder jsonBuilder = new StringBuilder();
            jsonBuilder.Append("{\"primary_bars\":[");

            // Send primary timeframe bars (BarsArray[0])
            int startBar = Math.Min(CurrentBars[0], historicalBarsToSend - 1);
            for (int barsAgo = startBar; barsAgo >= 0; barsAgo--)
            {
                if (barsAgo < startBar)
                    jsonBuilder.Append(",");

                jsonBuilder.AppendFormat("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5}}}",
                    Times[0][barsAgo].ToString("yyyy-MM-ddTHH:mm:ss"),
                    Opens[0][barsAgo],
                    Highs[0][barsAgo],
                    Lows[0][barsAgo],
                    Closes[0][barsAgo],
                    Volumes[0][barsAgo]);
            }

            jsonBuilder.Append("],\"secondary_bars\":[");

            // Send secondary timeframe bars (BarsArray[1] - 5-min)
            // Check if secondary bars are available
            if (BarsArray.Length > 1 && CurrentBars[1] >= 0)
            {
                int startBar5min = Math.Min(CurrentBars[1], historicalBarsToSend / 5);  // Fewer 5-min bars
                Print("SENDING SECONDARY DATA (" + (startBar5min + 1) + " bars of 5-min data)");

                for (int barsAgo = startBar5min; barsAgo >= 0; barsAgo--)
                {
                    if (barsAgo < startBar5min)
                        jsonBuilder.Append(",");

                    jsonBuilder.AppendFormat("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5}}}",
                        Times[1][barsAgo].ToString("yyyy-MM-ddTHH:mm:ss"),
                        Opens[1][barsAgo],
                        Highs[1][barsAgo],
                        Lows[1][barsAgo],
                        Closes[1][barsAgo],
                        Volumes[1][barsAgo]);
                }
            }
            else
            {
                Print("WARNING: Secondary timeframe (5-min) not available yet!");
            }

            jsonBuilder.AppendFormat("],\"type\":\"historical\",\"dailyGoal\":{0},\"dailyMaxLoss\":{1},\"contract\":\"{2}\"}}",
                dailyProfitGoal,
                dailyMaxLoss,
                Instrument.MasterInstrument.Name);
            string jsonData = jsonBuilder.ToString();

            Print("SENDING HISTORICAL DATA (" + barCount + " bars)");

            // Send asynchronously
            Task.Run(async () =>
            {
                try
                {
                    var content = new StringContent(jsonData, Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(serverUrl, content);

                    if (response.IsSuccessStatusCode)
                    {
                        Print("Historical data sent successfully to " + serverUrl);
                        serverConnected = true;
                        serverStatus = "Connected";
                    }
                    else
                    {
                        Print("Failed to send historical data. Status: " + response.StatusCode);
                        serverConnected = false;
                        serverStatus = "Error: " + response.StatusCode;
                    }
                }
                catch (Exception ex)
                {
                    Print("Error sending historical data: " + ex.Message);
                    serverConnected = false;
                    serverStatus = "Connection Failed";
                }
            });
        }

        private void SendBarData(int barIndex)
        {
            // Update daily P&L BEFORE sending data to server
            UpdateDailyPnL();

            // CHECK DAILY LIMITS - Don't send request if limits are hit
            if (dailyPnL >= dailyProfitGoal && dailyProfitGoal > 0)
            {
                Print($"[LIMIT CHECK] Daily profit goal reached ({dailyPnL:C2}) - NOT sending bar data");
                return;
            }

            if (dailyPnL <= -dailyMaxLoss && dailyMaxLoss > 0)
            {
                Print($"[LIMIT CHECK] Daily max loss reached ({dailyPnL:C2}) - NOT sending bar data");
                return;
            }

            // Capture primary timeframe bar data (BarsArray[0])
            string barTime = Times[0][0].ToString("yyyy-MM-ddTHH:mm:ss");
            double barOpen = Opens[0][0];
            double barHigh = Highs[0][0];
            double barLow = Lows[0][0];
            double barClose = Closes[0][0];
            double barVolume = Volumes[0][0];

            // Capture secondary timeframe bar data (BarsArray[1] - 5-min) if available
            string barTime5min = "";
            double barOpen5min = 0, barHigh5min = 0, barLow5min = 0, barClose5min = 0, barVolume5min = 0;
            bool hasSecondaryData = (BarsArray.Length > 1 && CurrentBars[1] >= 0);

            if (hasSecondaryData)
            {
                barTime5min = Times[1][0].ToString("yyyy-MM-ddTHH:mm:ss");
                barOpen5min = Opens[1][0];
                barHigh5min = Highs[1][0];
                barLow5min = Lows[1][0];
                barClose5min = Closes[1][0];
                barVolume5min = Volumes[1][0];
            }

            // Get current account balance from NinjaTrader
            double accountBalance = Account.Get(AccountItem.CashValue, Currency.UsDollar);

            // Get current position information for exit analysis
            string currentPosition = Position.MarketPosition == MarketPosition.Long ? "long" :
                                    Position.MarketPosition == MarketPosition.Short ? "short" : "flat";
            double entryPrice = Position.AveragePrice;
            int positionQuantity = Position.Quantity;

            Task.Run(async () =>
            {
                try
                {
                    string json;
                    if (hasSecondaryData)
                    {
                        json = string.Format("{{\"primary_bar\":{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5}}},\"secondary_bar\":{{\"time\":\"{6}\",\"open\":{7},\"high\":{8},\"low\":{9},\"close\":{10},\"volume\":{11}}},\"type\":\"realtime\",\"dailyPnL\":{12},\"dailyGoal\":{13},\"dailyMaxLoss\":{14},\"accountBalance\":{15},\"current_position\":\"{16}\",\"entry_price\":{17},\"position_quantity\":{18},\"contract\":\"{19}\"}}",
                        barTime,
                        barOpen,
                        barHigh,
                        barLow,
                        barClose,
                        barVolume,
                        barTime5min,
                        barOpen5min,
                        barHigh5min,
                        barLow5min,
                        barClose5min,
                        barVolume5min,
                        dailyPnL,
                        dailyProfitGoal,
                        dailyMaxLoss,
                        accountBalance,
                        currentPosition,
                        entryPrice,
                        positionQuantity,
                        Instrument.MasterInstrument.Name);
                    }
                    else
                    {
                        // Legacy format without secondary data
                        json = string.Format("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5},\"type\":\"realtime\",\"dailyPnL\":{6},\"dailyGoal\":{7},\"dailyMaxLoss\":{8},\"accountBalance\":{9},\"current_position\":\"{10}\",\"entry_price\":{11},\"position_quantity\":{12},\"contract\":\"{13}\"}}",
                            barTime,
                            barOpen,
                            barHigh,
                            barLow,
                            barClose,
                            barVolume,
                            dailyPnL,
                            dailyProfitGoal,
                            dailyMaxLoss,
                            accountBalance,
                            currentPosition,
                            entryPrice,
                            positionQuantity,
                            Instrument.MasterInstrument.Name);
                    }

                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(serverUrl, content);

                    if (response.IsSuccessStatusCode)
                    {
                        // Parse the response to get trade signal
                        string responseBody = await response.Content.ReadAsStringAsync();

                        // Parse JSON response
                        dynamic result = Newtonsoft.Json.JsonConvert.DeserializeObject(responseBody);

                        if (result.signal != null)
                        {
                            string signal = result.signal.ToString().ToLower();
                            double confidence = (double)result.confidence;

                            // Extract risk management parameters from server response
                            double stopLoss = 0.0;
                            double takeProfit = 0.0;

                            if (result.risk_management != null)
                            {
                                stopLoss = (double)result.risk_management.stop_loss;
                                takeProfit = (double)result.risk_management.take_profit;

                                Print("Risk Management - SL: " + stopLoss.ToString("F2") + ", TP: " + takeProfit.ToString("F2"));
                            }

                            // Check for early exit flag
                            bool earlyExitTriggered = false;
                            string exitReason = "";
                            if (result.exit_analysis != null && result.exit_analysis.early_exit_triggered != null)
                            {
                                earlyExitTriggered = (bool)result.exit_analysis.early_exit_triggered;
                                if (earlyExitTriggered && result.exit_analysis.reason != null)
                                {
                                    exitReason = result.exit_analysis.reason.ToString();
                                }
                            }

                            Print("AI Signal: " + signal.ToUpper() + " (" + (confidence * 100).ToString("F2") + "%)");
                            if (earlyExitTriggered)
                            {
                                Print("  ** EARLY EXIT DETECTED: " + exitReason);
                            }

                            // Execute trades based on signal and trading settings
                            ExecuteTradeSignal(signal, confidence, stopLoss, takeProfit, earlyExitTriggered, exitReason);
                        }
                    }
                    else
                    {
                        Print("Failed to send bar data. Status: " + response.StatusCode);
                    }
                }
                catch (Exception ex)
                {
                    Print("Error sending bar data: " + ex.Message);
                }
            });
        }

        private void ExecuteTradeSignal(string signal, double confidence, double stopLoss, double takeProfit, bool earlyExitTriggered = false, string exitReason = "")
        {
            // Update dashboard state with new signal
            lastSignal = signal;
            lastConfidence = confidence;
            lastSignalTime = DateTime.Now;
            totalSignals++;

            if (signal == "long")
                longSignals++;
            else if (signal == "short")
                shortSignals++;
            else if (signal == "hold")
                holdSignals++;

            // Only execute if trading is enabled
            if (!tradingEnabled)
            {
                Print("Trading is disabled - skipping signal");
                return;
            }

            // Execute on UI thread
            ChartControl.Dispatcher.InvokeAsync(() =>
            {
                try
                {
                    if (signal == "long" && allowLongTrades)
                    {
                        if (Position.MarketPosition == MarketPosition.Flat && !isLongOrderPending)
                        {
                            // Enter new long position with SL/TP
                            Print("LONG SIGNAL - Entering 1 contract");
                            Print("  Stop Loss: " + stopLoss.ToString("F2") + ", Take Profit: " + takeProfit.ToString("F2"));
                            desiredPosition = "flat";
                            isLongOrderPending = true;
                            isShortOrderPending = false;

                            // Set stop loss and take profit if provided by server
                            if (stopLoss > 0 && takeProfit > 0)
                            {
                                SetStopLoss("AILong", CalculationMode.Price, stopLoss, false);
                                SetProfitTarget("AILong", CalculationMode.Price, takeProfit);
                            }

                            EnterLong(1, "AILong");
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // OPPOSITE SIGNAL - Reverse from short to long
                            Print($"LONG SIGNAL - Reversing from SHORT (Closing {Position.Quantity} contracts)");
                            desiredPosition = "long";  // Set desired position
                            desiredStopLoss = stopLoss;  // Store SL for delayed entry
                            desiredTakeProfit = takeProfit;  // Store TP for delayed entry
                            isShortOrderPending = false;
                            ExitShort();  // Exit current short, then OnExecutionUpdate will enter long
                        }
                        else if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // Already long - stay in position (trend following)
                            Print($"LONG SIGNAL - Already in LONG position ({Position.Quantity} contracts) - Holding");

                            // Verify it's only 1 contract
                            if (Position.Quantity > 1)
                            {
                                Print($"WARNING: LONG position has {Position.Quantity} contracts, reducing to 1");
                                ExitLong(Position.Quantity - 1, "AILong", "AILong");
                            }
                        }
                    }
                    else if (signal == "short" && allowShortTrades)
                    {
                        if (Position.MarketPosition == MarketPosition.Flat && !isShortOrderPending)
                        {
                            // Enter new short position with SL/TP
                            Print("SHORT SIGNAL - Entering 1 contract");
                            Print("  Stop Loss: " + stopLoss.ToString("F2") + ", Take Profit: " + takeProfit.ToString("F2"));
                            desiredPosition = "flat";
                            isShortOrderPending = true;
                            isLongOrderPending = false;

                            // Set stop loss and take profit if provided by server
                            if (stopLoss > 0 && takeProfit > 0)
                            {
                                SetStopLoss("AIShort", CalculationMode.Price, stopLoss, false);
                                SetProfitTarget("AIShort", CalculationMode.Price, takeProfit);
                            }

                            EnterShort(1, "AIShort");
                        }
                        else if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // OPPOSITE SIGNAL - Reverse from long to short
                            Print($"SHORT SIGNAL - Reversing from LONG (Closing {Position.Quantity} contracts)");
                            desiredPosition = "short";  // Set desired position
                            desiredStopLoss = stopLoss;  // Store SL for delayed entry
                            desiredTakeProfit = takeProfit;  // Store TP for delayed entry
                            isLongOrderPending = false;
                            ExitLong();  // Exit current long, then OnExecutionUpdate will enter short
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // Already short - stay in position (trend following)
                            Print($"SHORT SIGNAL - Already in SHORT position ({Position.Quantity} contracts) - Holding");

                            // Verify it's only 1 contract
                            if (Position.Quantity > 1)
                            {
                                Print($"WARNING: SHORT position has {Position.Quantity} contracts, reducing to 1");
                                ExitShort(Position.Quantity - 1, "AIShort", "AIShort");
                            }
                        }
                    }
                    else if (signal == "hold")
                    {
                        // ====================================================================
                        // HOLD SIGNAL HANDLING - Two Scenarios:
                        // ====================================================================
                        // 1. EARLY EXIT (earlyExitTriggered = true)
                        //    - Intelligent exit condition detected by server
                        //    - Exit position immediately
                        //
                        // 2. REGULAR HOLD (earlyExitTriggered = false)
                        //    - Model says "no strong signal"
                        //    - Stay in position (trend following)
                        //    - Wait for opposite signal, stop loss, or take profit
                        // ====================================================================

                        if (earlyExitTriggered)
                        {
                            // EARLY EXIT - Exit position now
                            desiredPosition = "flat";

                            if (Position.MarketPosition == MarketPosition.Long)
                            {
                                Print($"🚨 EARLY EXIT - Closing LONG position ({Position.Quantity} contracts)");
                                Print($"  Reason: {exitReason}");
                                ExitLong();
                            }
                            else if (Position.MarketPosition == MarketPosition.Short)
                            {
                                Print($"🚨 EARLY EXIT - Closing SHORT position ({Position.Quantity} contracts)");
                                Print($"  Reason: {exitReason}");
                                ExitShort();
                            }
                            else
                            {
                                Print("🚨 EARLY EXIT signal received but no position open");
                            }
                        }
                        else
                        {
                            // REGULAR HOLD - Stay in position (trend following)
                            if (Position.MarketPosition == MarketPosition.Long)
                            {
                                Print($"HOLD SIGNAL - Staying in LONG position ({Position.Quantity} contracts)");
                                Print("  Exit Strategy: Will exit on SHORT signal, stop loss, take profit, or early exit");
                            }
                            else if (Position.MarketPosition == MarketPosition.Short)
                            {
                                Print($"HOLD SIGNAL - Staying in SHORT position ({Position.Quantity} contracts)");
                                Print("  Exit Strategy: Will exit on LONG signal, stop loss, take profit, or early exit");
                            }
                            else
                            {
                                Print("HOLD SIGNAL - No position, staying flat");
                            }

                            // Do nothing - let the position run
                            // Exits will happen via:
                            // 1. Stop Loss hit
                            // 2. Take Profit hit
                            // 3. Opposite signal (LONG->SHORT or SHORT->LONG)
                            // 4. Early exit detection (next bar)
                        }
                    }
                }
                catch (Exception ex)
                {
                    Print("Error executing trade: " + ex.Message);
                }
            });
        }






        #region Properties
        [NinjaScriptProperty]
        [Range(100, 50000)]
        [Display(Name = "Historical Bars To Send", Description = "Number of historical bars to send to RNN server", Order = 1, GroupName = "AI Parameters")]
        public int HistoricalBarsToSend
        {
            get { return historicalBarsToSend; }
            set { historicalBarsToSend = value; }
        }

        [NinjaScriptProperty]
        [Range(0, 10000)]
        [Display(Name = "Daily Profit Goal", Description = "Daily profit goal in currency", Order = 1, GroupName = "Risk Management")]
        public double DailyProfitGoal
        {
            get { return dailyProfitGoal; }
            set { dailyProfitGoal = value; }
        }

        [NinjaScriptProperty]
        [Range(0, 10000)]
        [Display(Name = "Daily Max Loss", Description = "Daily maximum loss in currency", Order = 2, GroupName = "Risk Management")]
        public double DailyMaxLoss
        {
            get { return dailyMaxLoss; }
            set { dailyMaxLoss = value; }
        }
        #endregion
    }
}
