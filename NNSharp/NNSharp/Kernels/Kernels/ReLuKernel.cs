using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels.Kernels
{
    public class ReLuKernel : IKernel
    {
        public void Execute()
        {
            input.ApplyToAll(p => { return Math.Max(0.0, p); });
            output = input;
        }

        public bool IsInplace()
        {
            return true;
        }

        protected IData input;
        protected IData output;
    }
}
