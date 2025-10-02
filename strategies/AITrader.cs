using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Xml.Serialization;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
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
        private string serverUrl = "https://df6159faef89.ngrok-free.app/analysis"; // Change to "http://YOUR_LOCAL_IP:8000/analysis" for VPS access
        private bool historicalDataSent = false;
        private int historicalBarsToSend = 5000; // Configurable parameter

        // Trading direction flags
        private bool allowLongTrades = true;
        private bool allowShortTrades = true;
        private bool tradingEnabled = true;

        // Chart Trader UI elements
        private Chart chartWindow;
        private Grid chartTraderGrid;
        private Button longOnlyButton;
        private Button shortOnlyButton;
        private Button bothDirectionsButton;
        private Button stopTradingButton;
        private Grid buttonGrid;
        private bool areButtonsAdded = false;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Description = "AI Trading Strategy with External Server Communication";
                Name = "AITrader";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
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
            }
            else if (State == State.Configure)
            {
                // Initialize HTTP client
                httpClient = new HttpClient();
                httpClient.Timeout = TimeSpan.FromSeconds(30);
            }
            else if (State == State.DataLoaded)
            {
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

                // Remove buttons
                if (ChartControl != null)
                {
                    ChartControl.Dispatcher.InvokeAsync(() => RemoveButtonsFromChartTrader());
                }
            }
        }

        protected override void OnBarUpdate()
        {
            // Wait until we have enough bars loaded before sending historical data
            if (!historicalDataSent)
            {
                // Check if we have enough bars OR if we're in realtime (use whatever bars are available)
                if (CurrentBar >= (historicalBarsToSend - 1) || State == State.Realtime)
                {
                    Print("===========================================");
                    Print("SENDING HISTORICAL DATA");
                    Print("Current Bar: " + CurrentBar);
                    Print("Bars to send: " + Math.Min(CurrentBar + 1, historicalBarsToSend));
                    Print("===========================================");
                    SendHistoricalData();
                    historicalDataSent = true;
                }
                else
                {
                    // Not enough bars yet, keep waiting
                    if (CurrentBar % 100 == 0) // Log every 100 bars
                    {
                        Print("Waiting for historical data... Current bars: " + (CurrentBar + 1) + " / " + historicalBarsToSend);
                    }
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
            int barCount = Math.Min(CurrentBar + 1, historicalBarsToSend);
            StringBuilder jsonBuilder = new StringBuilder();
            jsonBuilder.Append("{\"bars\":[");

            // Send only the requested number of bars (or all available if less)
            int startBar = Math.Min(CurrentBar, historicalBarsToSend - 1);
            for (int barsAgo = startBar; barsAgo >= 0; barsAgo--)
            {
                if (barsAgo < startBar)
                    jsonBuilder.Append(",");

                jsonBuilder.AppendFormat("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4}}}",
                    Time[barsAgo].ToString("yyyy-MM-ddTHH:mm:ss"),
                    Open[barsAgo],
                    High[barsAgo],
                    Low[barsAgo],
                    Close[barsAgo]);
            }

            jsonBuilder.Append("],\"type\":\"historical\"}");
            string jsonData = jsonBuilder.ToString();

            // Log the historical data being sent
            Print("========================================");
            Print("SENDING HISTORICAL DATA (" + barCount + " bars)");
            Print("========================================");
            Print(jsonData);
            Print("========================================");

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
                    }
                    else
                    {
                        Print("Failed to send historical data. Status: " + response.StatusCode);
                    }
                }
                catch (Exception ex)
                {
                    Print("Error sending historical data: " + ex.Message);
                }
            });
        }

        private void SendBarData(int barIndex)
        {
            // Capture bar data before async operation to prevent race conditions
            string barTime = Time[0].ToString("yyyy-MM-ddTHH:mm:ss");
            double barOpen = Open[0];
            double barHigh = High[0];
            double barLow = Low[0];
            double barClose = Close[0];

            Task.Run(async () =>
            {
                try
                {
                    string json = string.Format("{{\"time\":\"{0}\",\"open\":{1},\"high\":{2},\"low\":{3},\"close\":{4},\"type\":\"realtime\"}}",
                        barTime,
                        barOpen,
                        barHigh,
                        barLow,
                        barClose);

                    // Log the new bar data being sent
                    Print("========================================");
                    Print("SENDING NEW BAR DATA");
                    Print("========================================");
                    Print(json);
                    Print("========================================");

                    var content = new StringContent(json, Encoding.UTF8, "application/json");
                    var response = await httpClient.PostAsync(serverUrl, content);

                    if (response.IsSuccessStatusCode)
                    {
                        // Parse the response to get trade signal
                        string responseBody = await response.Content.ReadAsStringAsync();
                        Print("Response: " + responseBody);

                        // Parse JSON response
                        dynamic result = Newtonsoft.Json.JsonConvert.DeserializeObject(responseBody);

                        if (result.signal != null)
                        {
                            string signal = result.signal.ToString().ToLower();
                            double confidence = (double)result.confidence;

                            Print("========================================");
                            Print("AI SIGNAL RECEIVED");
                            Print("========================================");
                            Print("Signal: " + signal.ToUpper());
                            Print("Confidence: " + (confidence * 100).ToString("F2") + "%");
                            Print("========================================");

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
            {
                Print("Trading disabled - signal ignored");
                return;
            }

            // Execute on UI thread
            ChartControl.Dispatcher.InvokeAsync(() =>
            {
                try
                {
                    if (signal == "long" && allowLongTrades)
                    {
                        // Enter long if we're not already in a position
                        if (Position.MarketPosition == MarketPosition.Flat)
                        {
                            Print("EXECUTING LONG ENTRY - Confidence: " + (confidence * 100).ToString("F2") + "%");
                            EnterLong();
                        }
                        else if (Position.MarketPosition == MarketPosition.Short)
                        {
                            // Exit short and enter long
                            Print("REVERSING FROM SHORT TO LONG - Confidence: " + (confidence * 100).ToString("F2") + "%");
                            ExitShort();
                            EnterLong();
                        }
                    }
                    else if (signal == "short" && allowShortTrades)
                    {
                        // Enter short if we're not already in a position
                        if (Position.MarketPosition == MarketPosition.Flat)
                        {
                            Print("EXECUTING SHORT ENTRY - Confidence: " + (confidence * 100).ToString("F2") + "%");
                            EnterShort();
                        }
                        else if (Position.MarketPosition == MarketPosition.Long)
                        {
                            // Exit long and enter short
                            Print("REVERSING FROM LONG TO SHORT - Confidence: " + (confidence * 100).ToString("F2") + "%");
                            ExitLong();
                            EnterShort();
                        }
                    }
                    else if (signal == "hold")
                    {
                        Print("AI recommends HOLD - No action taken");
                    }
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
        #endregion
    }
}
