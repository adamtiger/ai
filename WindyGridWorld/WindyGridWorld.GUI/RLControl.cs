using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace WindyGridWorld.GUI
{

    public class RLControl
    {

        public RLControl()
        {
            iter = 0;
        }

        public void StartRL(
            int type,                   // the type of the rl algorithm
            int rows, int columns,      // the size of the world
            int numofEpisodes,          // number of learning episodes
            int startX, int startY,     // coordinates of the starting cell
            int targetX, int targetY,   // coordinates of the target cell
            double alpha,               // the learning rate
            double gamma,               // in case of discounted reward
            out TraceContainer container // the epsiodes with the trace of the agent
            )
        {
            iter = 0;
            numEps = numofEpisodes;

            // TODO: CLI call (set the initial values)

            container = new TraceContainer(); 
            container.Add(0, 1, 2);
        }

        public double Learn(int nextNepisode, TraceContainer container)
        {
            // TODO: CLI calls (learn and copy the positions if necessary)
            Thread.Sleep(1000);

            iter += nextNepisode;

            return iter * 1.0 / numEps;
        }

        private int iter;
        private int numEps;
    }
}
