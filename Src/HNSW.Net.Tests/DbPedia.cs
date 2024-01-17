using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace HNSW.Net.Tests
{
    internal class DbPedia
    {
        public required string Id { get; set; }
        public required string Title { get; set; }
        public required string Text { get; set; }
        public required List<float> Embeddings { get; set; }
    }
}
