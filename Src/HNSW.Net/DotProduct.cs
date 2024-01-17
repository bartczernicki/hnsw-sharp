using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Numerics.Tensors;

namespace HNSW.Net
{
    public static class DotProductDistance
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProductOptimized(float[] lhs, float[] rhs)
        {
            return (1 - TensorPrimitives.Dot(lhs, rhs));
        }
    }
}
