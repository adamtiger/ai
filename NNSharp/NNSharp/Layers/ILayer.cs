using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Layers
{
    public interface ILayer
    {
        void Execute(Data input, out Data output);

        void Execute(Data input, out Data output1, out Data output2);

        void Execute(Data input1, Data input2, out Data output);

        bool IsInplace();
    }
}
