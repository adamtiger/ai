using Newtonsoft.Json.Linq;
using NNSharp.DataTypes;
using NNSharp.IO;
using NNSharp.KernelDescriptors;
using NNSharp.Models;
using NNSharp.SequentialBased.SequentialExecutors;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NNSharp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var reader = new ReaderKerasModel("test.json");

            SequentialModel model = reader.GetSequentialExecutor();

            Console.WriteLine("Finished.");
        }
    }
}
