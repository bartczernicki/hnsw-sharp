// <copyright file="SmallWorldTests.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Numerics.Tensors;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for <see cref="SmallWorld{TItem, TDistance}"/>
    /// </summary>
    [TestClass]
    public class SmallWorldTests
    {
        // Set floating point error to 9.00 * 10^-7
        // For cosine distance error can be bigger in theory but for test data it's not the case.
        private const float FloatError = 0.000000900f;

        private IReadOnlyList<float[]> embeddingsVectors;
        private IReadOnlyList<DbPedia> dbPedias;

        /// <summary>
        /// Initializes test resources.
        /// </summary>
        [TestInitialize]
        public void TestInitialize()
        {
            // load dbpedias.json file
            var data = File.ReadAllText(@"dbpedias.json");
            dbPedias = System.Text.Json.JsonSerializer.Deserialize<List<DbPedia>>(data);
            embeddingsVectors = dbPedias.Select(a => a.Embeddings.ToArray()).ToList();
        }

        /// <summary>
        /// Basic test for knn search - this test might fail sometimes, as the construction of the graph does not guarantee an exact answer
        /// </summary>
        [TestMethod]
        public void KNNSearchTest()
        {
            var parameters = new SmallWorld<float[], float>.Parameters();
            var graph = new SmallWorld<float[], float>(DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(embeddingsVectors);

            int bestWrong = 0;
            float maxError = float.MinValue;

            for (int i = 0; i < embeddingsVectors.Count; ++i)
            {
                var result = graph.KNNSearch(embeddingsVectors[i], 20);
                var best = result.OrderBy(r => r.Distance).First();
                Assert.AreEqual(20, result.Count);
                if (best.Id != i)
                {
                    bestWrong++;
                }
                maxError = Math.Max(maxError, best.Distance);
            }
            Assert.AreEqual(0, bestWrong);
            Assert.AreEqual(0, maxError, FloatError);
        }

        /// <summary>
        /// Basic test for knn search - this test might fail sometimes, as the construction of the graph does not guarantee an exact answer
        /// </summary>
        [DataTestMethod]
        [DataRow(true, true)]
        [DataRow(false, true)]
        //[DataRow(false, false)]
        //[DataRow(true, false)]
        public void KNNSearchTestAlgorithm2(bool expandBestSelection, bool keepPrunedConnections)
        {
            var parameters = new SmallWorld<float[], float>.Parameters() 
            {
                M = 10,
                LevelLambda = 1 / Math.Log(10), // should match M
                EnableDistanceCacheForConstruction = true,
                InitialItemsSize = 100,
                InitialDistanceCacheSize = 100 * 1024,
                NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, 
                ExpandBestSelection = expandBestSelection, 
                KeepPrunedConnections = keepPrunedConnections 
            };

            var graph = new SmallWorld<float[], float>(DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(embeddingsVectors);

            int bestWrong = 0;
            float maxError = float.MinValue;

            for (int i = 0; i < embeddingsVectors.Count; ++i)
            {
                var result = graph.KNNSearch(embeddingsVectors[i], 20);
                var best = result.OrderBy(r => r.Distance).First();
                Assert.AreEqual(20, result.Count);
                if (best.Id != i)
                {
                    bestWrong++;
                }
                maxError = Math.Max(maxError, best.Distance);
            }
            Assert.AreEqual(0, bestWrong);
            Assert.AreEqual(0, maxError, FloatError);
        }

        /// <summary>
        /// Serialization deserialization tests.
        /// </summary>
        [TestMethod]
        public void SerializeDeserializeTest()
        {
            //byte[] buffer;
            string original;

            // restrict scope of original graph
            var stream = new MemoryStream();
            {
                var mParameter = 15;
                var parameters = new SmallWorld<float[], float>.Parameters()
                {
                    M = mParameter,
                    LevelLambda = 1 / Math.Log(mParameter),
                };

                var graph = new SmallWorld<float[], float>(DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance, parameters);
                graph.AddItems(embeddingsVectors);

                graph.SerializeGraph(stream);
                original = graph.Print();
            }
            stream.Position = 0;

            var copy = SmallWorld<float[], float>.DeserializeGraph(embeddingsVectors, DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance, stream);

            Assert.AreEqual(original, copy.Print());
        }

        [TestMethod]
        public void CompareVectorDistanceMethods()
        {
            var cosineDistance = CosineDistance.NonOptimized(embeddingsVectors[0], embeddingsVectors[1]);

            var dotProductDistance = DotProduct.DotProductOptimized(embeddingsVectors[0], embeddingsVectors[1]);

            Assert.AreEqual(cosineDistance, dotProductDistance, FloatError);
        }

        [TestMethod]
        public void FindFirstVector()
        {
            float maxError = float.MinValue;

            var parameters = new SmallWorld<float[], float>.Parameters() { NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, ExpandBestSelection = true, KeepPrunedConnections = true };
            var graph = new SmallWorld<float[], float>(DotProduct.DotProductOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(embeddingsVectors);

            var searchResults = graph.KNNSearch(embeddingsVectors[0], 100);
            // Ensure 100 results are returned
            Assert.AreEqual(100, searchResults.Count);

            var topMatch = searchResults.OrderBy(r => r.Distance).First();
            maxError = Math.Max(maxError, topMatch.Distance);

            // Ensure the top match is the first vector
            Assert.AreEqual(0, maxError, FloatError);
        }
    }
}