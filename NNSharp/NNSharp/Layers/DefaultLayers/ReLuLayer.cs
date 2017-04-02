using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Layers.DefaultLayers
{
    public class ReLuLayer : ILayer
    {
        public void Execute(IData input, out IData output)
        {
            input.ApplyToAll(p => { return Math.Max(0.0, p); });
            output = input;
        }

        public void Execute(IData input1, IData input2, out IData output)
        {
            throw new NotImplementedException();
        }

        public void Execute(IData input, out IData output1, out IData output2)
        {
            throw new NotImplementedException();
        }

        public bool IsInplace()
        {
            return true;
        }
    }
}
