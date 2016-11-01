using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindyGridWorld.GUI
{
    public delegate void ProcessStatusChanged(double percentage);

    public class RLControl
    {
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
            container = new TraceContainer(); // todo
            container.Add(0, 1, 2);
            processStatusChanged(0.5);
        }

        public event ProcessStatusChanged processStatusChanged; 
    }
}
