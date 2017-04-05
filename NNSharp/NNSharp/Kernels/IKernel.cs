﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp.Kernels
{
    public interface IKernel
    {
        void Execute();

        bool IsInplace();
    }
}
