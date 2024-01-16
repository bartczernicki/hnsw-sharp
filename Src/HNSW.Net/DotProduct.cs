using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using System.Numerics.Tensors;

namespace HNSW.Net
{
    public static class DotProduct
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProductOptimized(float[] lhs, float[] rhs)
        {
            return TensorPrimitives.Dot(lhs, rhs);
        }
    }
}
