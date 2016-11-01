using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WindyGridWorld.GUI
{
    public class TraceContainer
    {

        public TraceContainer(int capacity = 1)
        {
            traceContainer = new Dictionary<int, List<Position>>(capacity);
            IsEmpty = true;
        }


        public void Add(int episodeId, int x, int y)
        {
            Position p = new Position(x, y);

            if (traceContainer.ContainsKey(episodeId))
            {
                traceContainer[episodeId].Add(p);
            }
            else
            {
                List<Position> newTrace = new List<Position>();
                newTrace.Add(p);
                traceContainer.Add(episodeId, newTrace);
            }

            IsEmpty = false;
        }

        public int GetLength(int episodeId) { return traceContainer[episodeId].Count; }

        public int GetX(int episodeId, int idx)
        {
            return traceContainer[episodeId][idx].X;
        }

        public int GetY(int episodeId, int idx)
        {
            return traceContainer[episodeId][idx].Y;
        }

        public bool IsEmpty { get; private set; }

        /** Dictionary: episodeId and position list pair. */
        private Dictionary<int, List<Position>> traceContainer;

        #region(Struct: Position)

        private struct Position
        {
            
            public Position(int x, int y)
            {
                X = x;
                Y = y;
            }

            public int X { get; private set; }
            public int Y { get; private set; }
        }

        #endregion
    }
}
