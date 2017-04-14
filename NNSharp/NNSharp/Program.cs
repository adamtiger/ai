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
            var reader = new ReaderKerasModel("tests/test_flat_model.json");

            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(4, 5, 2, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 1, 1, 0] = 1;
            inp[0, 2, 0, 0] = 0;
            inp[0, 2, 1, 0] = 0;
            inp[0, 3, 0, 0] = 2;
            inp[0, 3, 1, 0] = 1;
            inp[0, 4, 0, 0] = 2;
            inp[0, 4, 1, 0] = 1;


            inp[1, 0, 0, 0] = 0;
            inp[1, 0, 1, 0] = -1;
            inp[1, 1, 0, 0] = 1;
            inp[1, 1, 1, 0] = -2;
            inp[1, 2, 0, 0] = 3;
            inp[1, 2, 1, 0] = 1;
            inp[1, 3, 0, 0] = 2;
            inp[1, 3, 1, 0] = 0;
            inp[1, 4, 0, 0] = 2;
            inp[1, 4, 1, 0] = -3;


            inp[2, 0, 0, 0] = 1;
            inp[2, 0, 1, 0] = 2;
            inp[2, 1, 0, 0] = -2;
            inp[2, 1, 1, 0] = 0;
            inp[2, 2, 0, 0] = 3;
            inp[2, 2, 1, 0] = -3;
            inp[2, 3, 0, 0] = 2;
            inp[2, 3, 1, 0] = 1;
            inp[2, 4, 0, 0] = 2;
            inp[2, 4, 1, 0] = 0;


            inp[3, 0, 0, 0] = 1;
            inp[3, 0, 1, 0] = 2;
            inp[3, 1, 0, 0] = 0;
            inp[3, 1, 1, 0] = -2;
            inp[3, 2, 0, 0] = 3;
            inp[3, 2, 1, 0] = 1;
            inp[3, 3, 0, 0] = 2;
            inp[3, 3, 1, 0] = 3;
            inp[3, 4, 0, 0] = -3;
            inp[3, 4, 1, 0] = 1;

            IData ou = model.ExecuteNetwork(inp);

            Console.WriteLine("Finished.");
        }

        private static void TestConv1()
        {
            var reader = new ReaderKerasModel("tests/test_conv_1_model.json");

            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(4, 5, 2, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 1, 1, 0] = 1;
            inp[0, 2, 0, 0] = 0;
            inp[0, 2, 1, 0] = 0;
            inp[0, 3, 0, 0] = 2;
            inp[0, 3, 1, 0] = 1;
            inp[0, 4, 0, 0] = 2;
            inp[0, 4, 1, 0] = 1;


            inp[1, 0, 0, 0] = 0;
            inp[1, 0, 1, 0] = -1;
            inp[1, 1, 0, 0] = 1;
            inp[1, 1, 1, 0] = -2;
            inp[1, 2, 0, 0] = 3;
            inp[1, 2, 1, 0] = 1;
            inp[1, 3, 0, 0] = 2;
            inp[1, 3, 1, 0] = 0;
            inp[1, 4, 0, 0] = 2;
            inp[1, 4, 1, 0] = -3;


            inp[2, 0, 0, 0] = 1;
            inp[2, 0, 1, 0] = 2;
            inp[2, 1, 0, 0] = -2;
            inp[2, 1, 1, 0] = 0;
            inp[2, 2, 0, 0] = 3;
            inp[2, 2, 1, 0] = -3;
            inp[2, 3, 0, 0] = 2;
            inp[2, 3, 1, 0] = 1;
            inp[2, 4, 0, 0] = 2;
            inp[2, 4, 1, 0] = 0;


            inp[3, 0, 0, 0] = 1;
            inp[3, 0, 1, 0] = 2;
            inp[3, 1, 0, 0] = 0;
            inp[3, 1, 1, 0] = -2;
            inp[3, 2, 0, 0] = 3;
            inp[3, 2, 1, 0] = 1;
            inp[3, 3, 0, 0] = 2;
            inp[3, 3, 1, 0] = 3;
            inp[3, 4, 0, 0] = -3;
            inp[3, 4, 1, 0] = 1;

            IData ou = model.ExecuteNetwork(inp);
        }

        private static void TestConv2()
        {
            var reader = new ReaderKerasModel("tests/test_conv_2_model.json");

            SequentialModel model = reader.GetSequentialExecutor();

            Data2D inp = new Data2D(4, 5, 2, 1);

            inp[0, 0, 0, 0] = 0;
            inp[0, 0, 1, 0] = 1;
            inp[0, 1, 0, 0] = 2;
            inp[0, 1, 1, 0] = 1;
            inp[0, 2, 0, 0] = 0;
            inp[0, 2, 1, 0] = 0;
            inp[0, 3, 0, 0] = 2;
            inp[0, 3, 1, 0] = 1;
            inp[0, 4, 0, 0] = 2;
            inp[0, 4, 1, 0] = 1;


            inp[1, 0, 0, 0] = 0;
            inp[1, 0, 1, 0] = -1;
            inp[1, 1, 0, 0] = 1;
            inp[1, 1, 1, 0] = -2;
            inp[1, 2, 0, 0] = 3;
            inp[1, 2, 1, 0] = 1;
            inp[1, 3, 0, 0] = 2;
            inp[1, 3, 1, 0] = 0;
            inp[1, 4, 0, 0] = 2;
            inp[1, 4, 1, 0] = -3;


            inp[2, 0, 0, 0] = 1;
            inp[2, 0, 1, 0] = 2;
            inp[2, 1, 0, 0] = -2;
            inp[2, 1, 1, 0] = 0;
            inp[2, 2, 0, 0] = 3;
            inp[2, 2, 1, 0] = -3;
            inp[2, 3, 0, 0] = 2;
            inp[2, 3, 1, 0] = 1;
            inp[2, 4, 0, 0] = 2;
            inp[2, 4, 1, 0] = 0;


            inp[3, 0, 0, 0] = 1;
            inp[3, 0, 1, 0] = 2;
            inp[3, 1, 0, 0] = 0;
            inp[3, 1, 1, 0] = -2;
            inp[3, 2, 0, 0] = 3;
            inp[3, 2, 1, 0] = 1;
            inp[3, 3, 0, 0] = 2;
            inp[3, 3, 1, 0] = 3;
            inp[3, 4, 0, 0] = -3;
            inp[3, 4, 1, 0] = 1;

            IData ou = model.ExecuteNetwork(inp);
        }
    }
}
