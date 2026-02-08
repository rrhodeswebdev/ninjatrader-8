using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.NinjaScript;
using SharpDX;
using SharpDX.Direct2D1;

namespace NinjaTrader.NinjaScript.Indicators.WolfToolz
{
	public class IntervalLines : Indicator
	{
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description					= "Horizontal lines at specified intervals above and below zero";
				Name						= "IntervalLines";
				Calculate					= Calculate.OnBarClose;
				IsOverlay					= true;
				DrawOnPricePanel			= true;
				DisplayInDataBox			= false;
				IsSuspendedWhileInactive	= true;

				Interval					= 1000;
				Levels						= 5;
				ShowNeutralLine				= true;
				PositiveColor				= Brushes.DodgerBlue;
				NegativeColor				= Brushes.Red;
				NeutralColor				= Brushes.Gray;
				LineThickness				= 1;
				LineOpacity					= 60;

				AddPlot(new Stroke(Brushes.Transparent, 0), PlotStyle.Line, "IntervalLines");
			}
		}

		protected override void OnBarUpdate() { }

		protected override void OnRender(ChartControl chartControl, ChartScale chartScale)
		{
			if (ChartBars == null)
				return;

			float xStart	= chartControl.GetXByBarIndex(ChartBars, ChartBars.FromIndex);
			float xEnd		= chartControl.GetXByBarIndex(ChartBars, ChartBars.ToIndex);
			float thickness	= (float)LineThickness;
			float opacity	= (float)LineOpacity / 100f;

			if (ShowNeutralLine)
				DrawHorizontalLine(chartScale, xStart, xEnd, 0, NeutralColor, thickness, opacity);

			for (int i = 1; i <= Levels; i++)
			{
				double value = i * Interval;
				DrawHorizontalLine(chartScale, xStart, xEnd, value, PositiveColor, thickness, opacity);
				DrawHorizontalLine(chartScale, xStart, xEnd, -value, NegativeColor, thickness, opacity);
			}
		}

		private void DrawHorizontalLine(ChartScale chartScale, float xStart, float xEnd, double value, System.Windows.Media.Brush color, float thickness, float opacity)
		{
			float y = chartScale.GetYByValue(value);

			using (var brush = color.ToDxBrush(RenderTarget))
			{
				brush.Opacity = opacity;
				RenderTarget.DrawLine(
					new Vector2(xStart, y),
					new Vector2(xEnd, y),
					brush, thickness);
			}
		}

		#region Properties

		[NinjaScriptProperty]
		[Range(0.01, double.MaxValue)]
		[Display(Name = "Interval", Description = "Distance between each horizontal line", Order = 1, GroupName = "Parameters")]
		public double Interval { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name = "Levels", Description = "Number of lines above and below zero", Order = 2, GroupName = "Parameters")]
		public int Levels { get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Show Zero Line", Description = "Display the neutral line at zero", Order = 3, GroupName = "Parameters")]
		public bool ShowNeutralLine { get; set; }

		[XmlIgnore]
		[Display(Name = "Positive Color", Description = "Color for lines above zero", Order = 1, GroupName = "Visual")]
		public System.Windows.Media.Brush PositiveColor { get; set; }

		[Browsable(false)]
		public string PositiveColorSerialize
		{
			get { return Serialize.BrushToString(PositiveColor); }
			set { PositiveColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Negative Color", Description = "Color for lines below zero", Order = 2, GroupName = "Visual")]
		public System.Windows.Media.Brush NegativeColor { get; set; }

		[Browsable(false)]
		public string NegativeColorSerialize
		{
			get { return Serialize.BrushToString(NegativeColor); }
			set { NegativeColor = Serialize.StringToBrush(value); }
		}

		[XmlIgnore]
		[Display(Name = "Neutral Color", Description = "Color for the zero line", Order = 3, GroupName = "Visual")]
		public System.Windows.Media.Brush NeutralColor { get; set; }

		[Browsable(false)]
		public string NeutralColorSerialize
		{
			get { return Serialize.BrushToString(NeutralColor); }
			set { NeutralColor = Serialize.StringToBrush(value); }
		}

		[NinjaScriptProperty]
		[Range(1, 10)]
		[Display(Name = "Line Thickness", Description = "Thickness of the horizontal lines", Order = 4, GroupName = "Visual")]
		public int LineThickness { get; set; }

		[NinjaScriptProperty]
		[Range(1, 100)]
		[Display(Name = "Line Opacity (%)", Description = "Opacity of the lines from 1 to 100 percent", Order = 5, GroupName = "Visual")]
		public int LineOpacity { get; set; }

		#endregion
	}
}
