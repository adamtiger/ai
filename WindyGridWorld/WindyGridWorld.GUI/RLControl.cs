using native.wrapper;
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
        // Wrapper class from CLI
        LearningFrameworkWrapper lfw;

        public RLControl()
        {
            lfw = new LearningFrameworkWrapper();
        }

        public void StartRL(
            int type,                    // the type of the rl algorithm
            int rows, int columns,       // the size of the world
            int numofEpisodes,           // number of learning episodes
            int startX, int startY,      // coordinates of the starting cell
            int targetX, int targetY,    // coordinates of the target cell
            double alpha,                // the learning rate
            double gamma,                // in case of discounted reward
            out TraceContainer container // the epsiodes with the trace of the agent
            )
        {
            numEps = numofEpisodes;

            lfw.InitFramework(
                type, rows, columns, numofEpisodes, startX, startY,
                targetX, targetY, alpha, gamma);

            container = new TraceContainer(); 
        }

        public double Learn(TraceContainer container)
        {
            lfw.Learn();

            int epsId = lfw.GetEpisodeId();
            int k = lfw.GetPathLength();
            for (int idx = 0; idx < lfw.GetPathLength(); ++idx)
            {
                int x = lfw.GetCoordX(idx);
                int y = lfw.GetCoordY(idx);

                container.Add(epsId, x, y);
            }

            return lfw.GetProgress();
        }

        private int numEps;
    }
}
