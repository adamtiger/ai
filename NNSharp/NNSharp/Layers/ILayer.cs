using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Layers
{
    public interface ILayer
    {
        void Execute(IData input, out IData output);

        void Execute(IData input, out IData output1, out IData output2);

        void Execute(IData input1, IData input2, out IData output);

        bool IsInplace();
    }
}
