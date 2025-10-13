using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
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
        private string serverUrl = "https://2f73a7f4d7fe.ngrok-free.app/analysis"; // Change to "http://YOUR_LOCAL_IP:8000/analysis" for VPS access
        private bool historicalDataSent = false;
        private int historicalBarsToSend = 2000; // Configurable parameter (1 week of 1-min bars)

        // Trading direction flags
        private bool allowLongTrades = true;
        private bool allowShortTrades = true;
        private bool tradingEnabled = true;

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

        // Chart Trader UI elements
        private Chart chartWindow;
        private Grid chartTraderGrid;
        private Button longOnlyButton;
        private Button shortOnlyButton;
        private Button bothDirectionsButton;
        private Button stopTradingButton;
        private Grid buttonGrid;
        private bool areButtonsAdded = false;

        // Dashboard state tracking
        private string lastSignal = "N/A";
        private double lastConfidence = 0.0;
        private DateTime lastSignalTime = DateTime.MinValue;
        private int totalSignals = 0;
        private int longSignals = 0;
        private int shortSignals = 0;
        private int holdSignals = 0;
        private bool serverConnected = false;
        private string serverStatus = "Connecting...";

        // Dashboard rendering
        private System.Windows.Media.Typeface dashboardTypeface;
        private System.Windows.Media.SolidColorBrush backgroundBrush;
        private System.Windows.Media.SolidColorBrush textBrush;
        private System.Windows.Media.SolidColorBrush headerBrush;
        private System.Windows.Media.SolidColorBrush longBrush;
        private System.Windows.Media.SolidColorBrush shortBrush;
        private System.Windows.Media.SolidColorBrush holdBrush;

        // Cached SharpDX objects for performance
        private SharpDX.DirectWrite.TextFormat cachedTextFormat;
        private bool needsChartRefresh = false;

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
                HistoricalBarsToSend = 5000;

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

                // Add buttons to Chart Trader after data is loaded
                if (ChartControl != null)
                {
                    ChartControl.Dispatcher.InvokeAsync(() => AddButtonsToChartTrader());
                }
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

                // Remove buttons
                if (ChartControl != null)
                {
                    ChartControl.Dispatcher.InvokeAsync(() => RemoveButtonsFromChartTrader());
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
                    desiredPosition = "flat";
                    isLongOrderPending = true;
                    EnterLong(1, "AILong");
                }
                else if (desiredPosition == "short" && allowShortTrades && tradingEnabled)
                {
                    Print("Position now FLAT - Entering SHORT as desired");
                    desiredPosition = "flat";
                    isShortOrderPending = true;
                    EnterShort(1, "AIShort");
                }
                else
                {
                    desiredPosition = "flat";
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
                Print("New trading day - Daily P&L reset");
            }

            // Add realized P&L from the account
            if (SystemPerformance != null && SystemPerformance.AllTrades.Count > 0)
            {
                // Calculate today's P&L from all trades
                dailyPnL = 0.0;
                foreach (var trade in SystemPerformance.AllTrades)
                {
                    if (trade.Exit.Time.Date == today)
                    {
                        dailyPnL += trade.ProfitCurrency;
                    }
                }
            }

            Print($"Daily P&L Updated: {dailyPnL:C2} (Goal: {dailyProfitGoal:C2}, Max Loss: -{dailyMaxLoss:C2})");

            // Check if we hit profit goal or max loss
            if (dailyPnL >= dailyProfitGoal)
            {
                Print($"*** DAILY PROFIT GOAL REACHED: {dailyPnL:C2} ***");
                tradingEnabled = false;
                UpdateButtonStates();
            }
            else if (dailyPnL <= -dailyMaxLoss)
            {
                Print($"*** DAILY MAX LOSS REACHED: {dailyPnL:C2} ***");
                tradingEnabled = false;
                UpdateButtonStates();

                // Exit any open positions
                if (Position.MarketPosition == MarketPosition.Long)
                    ExitLong();
                else if (Position.MarketPosition == MarketPosition.Short)
                    ExitShort();
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

            // Send current bar data on each bar close (real-time only)
            if (State == State.Realtime)
            {
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

            jsonBuilder.AppendFormat("],\"type\":\"historical\",\"dailyGoal\":{0},\"dailyMaxLoss\":{1}}}",
                dailyProfitGoal,
                dailyMaxLoss);
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

            Task.Run(async () =>
            {
                try
                {
                    string json;
                    if (hasSecondaryData)
                    {
                        json = string.Format("{{\"primary_bar\":{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5}}},\"secondary_bar\":{{\"time\":\"{6}\",\"open\":{7},\"high\":{8},\"low\":{9},\"close\":{10},\"volume\":{11}}},\"type\":\"realtime\",\"dailyPnL\":{12},\"dailyGoal\":{13},\"dailyMaxLoss\":{14},\"accountBalance\":{15}}}",
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
                        accountBalance);
                    }
                    else
                    {
                        // Legacy format without secondary data
                        json = string.Format("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"volume\":{5},\"type\":\"realtime\",\"dailyPnL\":{6},\"dailyGoal\":{7},\"dailyMaxLoss\":{8},\"accountBalance\":{9}}}",
                            barTime,
                            barOpen,
                            barHigh,
                            barLow,
                            barClose,
                            barVolume,
                            dailyPnL,
                            dailyProfitGoal,
                            dailyMaxLoss,
                            accountBalance);
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

                            // Update dashboard state
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

                            Print("AI Signal: " + signal.ToUpper() + " (" + (confidence * 100).ToString("F2") + "%)");

                            // Execute trades based on signal and trading settings
                            ExecuteTradeSignal(signal, confidence);
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

        private void ExecuteTradeSignal(string signal, double confidence)
        {
            // Only execute if trading is enabled
            if (!tradingEnabled)
                return;

            // Execute on UI thread
            ChartControl.Dispatcher.InvokeAsync(() =>
            {
                try
                {
                    if (signal == "long" && allowLongTrades)
                    {
                        if (Position.MarketPosition == MarketPosition.Flat && !isLongOrderPending)
                        {
                            // Enter new long position
                            Print("LONG SIGNAL - Entering 1 contract");
                            desiredPosition = "flat";
                            isLongOrderPending = true;
                            isShortOrderPending = false;
                            EnterLong(1, "AILong");
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // Need to reverse from short to long
                            Print($"LONG SIGNAL - Closing {Position.Quantity} SHORT contracts first");
                            desiredPosition = "long";  // Set desired position
                            isShortOrderPending = false;
                            ExitShort();  // Exit current short, then OnExecutionUpdate will enter long
                        }
                        else if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // Already long - verify it's only 1 contract
                            if (Position.Quantity > 1)
                            {
                                Print($"WARNING: LONG position has {Position.Quantity} contracts, reducing to 1");
                                ExitLong(Position.Quantity - 1, "AILong", "AILong");
                            }
                            // else already in correct position with 1 contract
                        }
                    }
                    else if (signal == "short" && allowShortTrades)
                    {
                        if (Position.MarketPosition == MarketPosition.Flat && !isShortOrderPending)
                        {
                            // Enter new short position
                            Print("SHORT SIGNAL - Entering 1 contract");
                            desiredPosition = "flat";
                            isShortOrderPending = true;
                            isLongOrderPending = false;
                            EnterShort(1, "AIShort");
                        }
                        else if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // Need to reverse from long to short
                            Print($"SHORT SIGNAL - Closing {Position.Quantity} LONG contracts first");
                            desiredPosition = "short";  // Set desired position
                            isLongOrderPending = false;
                            ExitLong();  // Exit current long, then OnExecutionUpdate will enter short
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // Already short - verify it's only 1 contract
                            if (Position.Quantity > 1)
                            {
                                Print($"WARNING: SHORT position has {Position.Quantity} contracts, reducing to 1");
                                ExitShort(Position.Quantity - 1, "AIShort", "AIShort");
                            }
                            // else already in correct position with 1 contract
                        }
                    }
                    else if (signal == "hold")
                    {
                        // Exit any open positions on HOLD signal
                        desiredPosition = "flat";
                        if (Position.MarketPosition == MarketPosition.Long)
                        {
                            Print($"HOLD SIGNAL - Closing {Position.Quantity} LONG contracts");
                            ExitLong();
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            Print($"HOLD SIGNAL - Closing {Position.Quantity} SHORT contracts");
                            ExitShort();
                        }
                    }
                    // HOLD signal - no action needed, already logged above
                }
                catch (Exception ex)
                {
                    Print("Error executing trade: " + ex.Message);
                }
            });
        }

        private void AddButtonsToChartTrader()
        {
            if (areButtonsAdded)
                return;

            try
            {
                // Get the chart window
                chartWindow = Window.GetWindow(ChartControl.Parent) as Chart;
                if (chartWindow == null)
                    return;

                // Find Chart Trader
                ChartTrader chartTrader = Window.GetWindow(chartWindow.ActiveChartControl.Parent)
                    .FindFirst("ChartWindowChartTraderControl") as ChartTrader;

                if (chartTrader == null)
                    return;

                // Get the main grid
                chartTraderGrid = chartTrader.FindName("grdMain") as Grid;
                if (chartTraderGrid == null)
                    return;

                // Add a new row for our buttons if needed
                if (chartTraderGrid.RowDefinitions.Count <= 7)
                    chartTraderGrid.RowDefinitions.Add(new RowDefinition() { Height = GridLength.Auto });

                // Create button style
                Style buttonStyle = Application.Current.TryFindResource("Button") as Style;

                // Create Long Only button
                longOnlyButton = new Button
                {
                    Content = "Longs Only",
                    Style = buttonStyle,
                    Margin = new Thickness(2),
                    Background = System.Windows.Media.Brushes.Green,
                    Foreground = System.Windows.Media.Brushes.White,
                    Height = 40,
                    FontSize = 14
                };
                longOnlyButton.Click += LongOnlyButton_Click;

                // Create Short Only button
                shortOnlyButton = new Button
                {
                    Content = "Shorts Only",
                    Style = buttonStyle,
                    Margin = new Thickness(2),
                    Background = System.Windows.Media.Brushes.Red,
                    Foreground = System.Windows.Media.Brushes.White,
                    Height = 40,
                    FontSize = 14
                };
                shortOnlyButton.Click += ShortOnlyButton_Click;

                // Create Both Directions button
                bothDirectionsButton = new Button
                {
                    Content = "Both",
                    Style = buttonStyle,
                    Margin = new Thickness(2),
                    Background = System.Windows.Media.Brushes.Blue,
                    Foreground = System.Windows.Media.Brushes.White,
                    Height = 40,
                    FontSize = 14
                };
                bothDirectionsButton.Click += BothDirectionsButton_Click;

                // Create Stop Trading button
                stopTradingButton = new Button
                {
                    Content = "Stop Trading",
                    Style = buttonStyle,
                    Margin = new Thickness(2),
                    Background = System.Windows.Media.Brushes.DarkGray,
                    Foreground = System.Windows.Media.Brushes.White,
                    Height = 40,
                    FontSize = 14
                };
                stopTradingButton.Click += StopTradingButton_Click;

                // Create grid to hold buttons (3 rows, 2 columns)
                buttonGrid = new Grid();
                buttonGrid.ColumnDefinitions.Add(new ColumnDefinition());
                buttonGrid.ColumnDefinitions.Add(new ColumnDefinition());
                buttonGrid.RowDefinitions.Add(new RowDefinition());
                buttonGrid.RowDefinitions.Add(new RowDefinition());
                buttonGrid.RowDefinitions.Add(new RowDefinition());

                // First row: Long and Short buttons
                Grid.SetColumn(longOnlyButton, 0);
                Grid.SetRow(longOnlyButton, 0);
                Grid.SetColumn(shortOnlyButton, 1);
                Grid.SetRow(shortOnlyButton, 0);

                // Second row: Both button spanning both columns
                Grid.SetColumn(bothDirectionsButton, 0);
                Grid.SetRow(bothDirectionsButton, 1);
                Grid.SetColumnSpan(bothDirectionsButton, 2);

                // Third row: Stop Trading button spanning both columns
                Grid.SetColumn(stopTradingButton, 0);
                Grid.SetRow(stopTradingButton, 2);
                Grid.SetColumnSpan(stopTradingButton, 2);

                buttonGrid.Children.Add(longOnlyButton);
                buttonGrid.Children.Add(shortOnlyButton);
                buttonGrid.Children.Add(bothDirectionsButton);
                buttonGrid.Children.Add(stopTradingButton);

                // Add button grid to Chart Trader
                Grid.SetRow(buttonGrid, 8);
                chartTraderGrid.Children.Add(buttonGrid);

                areButtonsAdded = true;
                UpdateButtonStates();
            }
            catch (Exception ex)
            {
                Print("Error adding buttons to Chart Trader: " + ex.Message);
            }
        }

        private void RemoveButtonsFromChartTrader()
        {
            if (!areButtonsAdded)
                return;

            try
            {
                if (longOnlyButton != null)
                {
                    longOnlyButton.Click -= LongOnlyButton_Click;
                    longOnlyButton = null;
                }

                if (shortOnlyButton != null)
                {
                    shortOnlyButton.Click -= ShortOnlyButton_Click;
                    shortOnlyButton = null;
                }

                if (bothDirectionsButton != null)
                {
                    bothDirectionsButton.Click -= BothDirectionsButton_Click;
                    bothDirectionsButton = null;
                }

                if (stopTradingButton != null)
                {
                    stopTradingButton.Click -= StopTradingButton_Click;
                    stopTradingButton = null;
                }

                if (chartTraderGrid != null && buttonGrid != null)
                {
                    chartTraderGrid.Children.Remove(buttonGrid);
                    buttonGrid = null;
                }

                areButtonsAdded = false;
            }
            catch (Exception ex)
            {
                Print("Error removing buttons from Chart Trader: " + ex.Message);
            }
        }

        private void LongOnlyButton_Click(object sender, RoutedEventArgs e)
        {
            allowLongTrades = true;
            allowShortTrades = false;
            tradingEnabled = true;
            Print("Trading mode: LONGS ONLY");
            UpdateButtonStates();
        }

        private void ShortOnlyButton_Click(object sender, RoutedEventArgs e)
        {
            allowLongTrades = false;
            allowShortTrades = true;
            tradingEnabled = true;
            Print("Trading mode: SHORTS ONLY");
            UpdateButtonStates();
        }

        private void BothDirectionsButton_Click(object sender, RoutedEventArgs e)
        {
            allowLongTrades = true;
            allowShortTrades = true;
            tradingEnabled = true;
            Print("Trading mode: BOTH DIRECTIONS");
            UpdateButtonStates();
        }

        private void StopTradingButton_Click(object sender, RoutedEventArgs e)
        {
            tradingEnabled = false;
            Print("Trading STOPPED");
            UpdateButtonStates();
        }

        private void UpdateButtonStates()
        {
            if (longOnlyButton != null && shortOnlyButton != null && bothDirectionsButton != null && stopTradingButton != null)
            {
                if (!tradingEnabled)
                {
                    // Trading is stopped - dim direction buttons, highlight stop button
                    longOnlyButton.Opacity = 0.3;
                    shortOnlyButton.Opacity = 0.3;
                    bothDirectionsButton.Opacity = 0.3;
                    stopTradingButton.Opacity = 1.0;
                }
                else
                {
                    // Trading is active - show normal states
                    stopTradingButton.Opacity = 0.5;

                    // Reset direction buttons to inactive state
                    longOnlyButton.Opacity = 0.5;
                    shortOnlyButton.Opacity = 0.5;
                    bothDirectionsButton.Opacity = 0.5;

                    // Highlight the active mode
                    if (allowLongTrades && allowShortTrades)
                    {
                        // Both directions active
                        bothDirectionsButton.Opacity = 1.0;
                    }
                    else if (allowLongTrades)
                    {
                        // Longs only
                        longOnlyButton.Opacity = 1.0;
                    }
                    else if (allowShortTrades)
                    {
                        // Shorts only
                        shortOnlyButton.Opacity = 1.0;
                    }
                }
            }
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
