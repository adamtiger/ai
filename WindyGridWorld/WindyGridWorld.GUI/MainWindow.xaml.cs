﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace WindyGridWorld.GUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();

            // List to store the traces of the agent.
            container = new TraceContainer();

            // Initialize the rlController.
            rlControl = new RLControl();
            rlControl.processStatusChanged += ProcessStatusChanged_RefreshBar;

            // Initialize variables.
            idx = 0;
            episode = 0;
        }

        #region(Functionalities of the controls)

        // Starting the reinforcement learning process.
        // Read the initializing data.
        private void startRL_Click(object sender, RoutedEventArgs e)
        {
            numRows = int.Parse(rows.Text);
            numCols = int.Parse(columns.Text);
            stX = int.Parse(startX.Text);
            stY = int.Parse(startY.Text);
            tgX = int.Parse(targetX.Text);
            tgY = int.Parse(targetY.Text);

            rlControl.StartRL(
                typeSelector.SelectedIndex, numRows, numCols,
                int.Parse(numEps.Text), 
                stX, stY,
                tgX, tgY, 
                double.Parse(alpha.Text), double.Parse(gamma.Text), 
                out container);
        }

        // Load the given episode.
        private void getEpsiode_Click(object sender, RoutedEventArgs e)
        {
            episode = int.Parse(episodeNum.Text);
        }

        // Handle the agent.
        // Set back the agent to the starting position.
        private void zeros_Click(object sender, RoutedEventArgs e)
        {
            idx = 0;
        }

        // Step to the next position on the grid.
        private void next_Click(object sender, RoutedEventArgs e)
        {
            idx += 1;
        }

        // Show the learning curve.
        private void learningCv_Click(object sender, RoutedEventArgs e)
        {

        }

        private void canvas_SizeChanged(object sender, SizeChangedEventArgs e)
        {
            RePaint();
        }

        #endregion

        #region(Helper functions)

        private void DrawAgent()
        {
            if (!container.IsEmpty)
            {
                double dX = w / numCols;
                double dY = h / numRows;

                int x = container.GetX(episode, idx);
                int y = container.GetY(episode, idx);

                Ellipse circ = new Ellipse();
                circ.Width = dX / 2;
                circ.Height = dY / 2;
                circ.Fill = Brushes.Red;

                Canvas.SetLeft(circ, x0 + dX * x + dX / 4);
                Canvas.SetTop(circ, y0 + dY * y + dY / 4);
                canvas.Children.Add(circ);
            }
        }

        private void PaintWindyGridWorld()
        {
            if (numRows > 0 && numCols > 0)
            {
                h = canvas.ActualHeight * 0.9; // 90% is used up from the canvas.
                w = canvas.ActualWidth * 0.9;

                y0 = (canvas.ActualHeight - h) / 2;
                x0 = (canvas.ActualWidth - w) / 2;

                double dX = w / numCols;
                double dY = h / numRows;

                canvas.Children.Clear();
                for (int c = 0; c < numCols + 2; ++c)
                {
                    Line line = new Line();
                    line.Stroke = Brushes.Black;

                    line.X1 = x0 + c * dX; line.Y1 = y0;
                    line.X2 = line.X1; line.Y2 = y0 + h;

                    canvas.Children.Add(line);
                }

                for (int r = 0; r < numRows + 2; ++r)
                {
                    Line line = new Line();
                    line.Stroke = Brushes.Black;

                    line.X1 = x0; line.Y1 = y0 + r * dY;
                    line.X2 = x0 + w; line.Y2 = line.Y1;

                    canvas.Children.Add(line);
                }

                Rectangle st = new Rectangle(), tg = new Rectangle();

                tg.Height = st.Height = dY - 1;
                tg.Width = st.Width = dX - 1;

                st.Fill = Brushes.Green;
                tg.Fill = Brushes.Blue;

                Canvas.SetLeft(st, x0 + stX * dX + 0.5);
                Canvas.SetTop(st, y0 + stY * dY + 0.5);

                Canvas.SetLeft(tg, x0 + tgX * dX + 0.5);
                Canvas.SetTop(tg, y0 + tgY * dY + 0.5);

                canvas.Children.Add(st);
                canvas.Children.Add(tg);
            }
        }

        private void RePaint()
        {
            PaintWindyGridWorld();
            for (int i = 0; i < idx + 1; ++i)
                DrawAgent();
        }

        #endregion

        private void ProcessStatusChanged_RefreshBar(double percentage)
        {
            progressBar.Value = percentage * 100;
        }

        #region(Variables)
        private TraceContainer container;
        private RLControl rlControl;
        private int episode;
        private int idx;

        private double x0, y0, h, w;

        private int numRows;
        private int numCols;
        private int stX;
        private int stY;
        private int tgX;
        private int tgY;
        #endregion
    }
}
