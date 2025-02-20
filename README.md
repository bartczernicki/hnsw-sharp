
<a href="https://curiosity.ai"><img src="https://curiosity.ai/media/cat.color.square.svg" width="100" height="100" align="right" /></a>


# HNSW.Net
.Net library for fast approximate nearest neighbours search using the HNSW (Hierarchical Navigitable Small Worlds) algorithm used in Pinecone, DataStax, Azure AI Search and other commercial vector databases.  

Exact _k_ nearest neighbours search algorithms tend to perform poorly in high-dimensional spaces. To overcome curse of dimensionality the ANN algorithms come in place. This library implements one of such algorithms described in the ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf) article. It provides simple API for building nearest neighbours graphs, (de)serializing them and running k-NN search queries.

## Benchmarks  
Benchmark search results from the https://github.com/bartczernicki/VectorMathAIOptimizations GitHub repository over 1 million real vectors from Wikipedia.  
Linear - Non-optimized benchmark using O(n) linear performance degredation  
Complete - Optimized benchmark with multi-threading, DotProduct, .NET 8 Tensors, AVX extensions using O(n) linear performance degredation  
CompleteRealDataANN - Optimized benchmark using the HNSW algorithm  
```
| Method              | Mean          | Error     | StdDev    | Ratio    | RatioSD | Search queries / second |
|-------------------- |--------------:|----------:|----------:|---------:|--------:|------------------------:|-
| Linear              | 1,664.7444 ms | 0.3980 ms | 0.3723 ms | baseline |         |       0.63 queries / sec|
| Complete            |    94.5493 ms | 1.8721 ms | 4.0299 ms |   -94.3% |    3.5% |      10.69 queries / sec| 
| CompleteRealDataANN |     0.6453 ms | 0.0012 ms | 0.0011 ms |  -100.0% |    0.2% |   1,550.39 queries / sec|
```

## Usage
Check out the following code snippets once you've added the library reference to your project.
##### How to build a HNSW graph?
```c#
var parameters = new SmallWorld<float[], float>.Parameters()
{
  M = 15, // defines the amount of neighbor connections for each vector, more connections create dense graphs with potentially higher recall
  LevelLambda = 1 / Math.Log(15),
};

float[] vectors = GetFloatVectors();
var graph = new SmallWorld<float[], float>(DotProductDistance.DotProductOptimized, DefaultRandomGenerator.Instance,
  parameters, threadSafe: true);
graph.AddItems(vectors);
```
##### How to run k-NN search in a HNSW graph?
```c#
SmallWorld<float[], float> graph = GetGraph();

float[] query = Enumerable.Repeat(1f, 100).ToArray();
var best20 = graph.KNNSearch(query, 20);
var best1 = best20.OrderBy(r => r.Distance).First();
```
##### How to (de)serialize the HNSW graph?
```c#
SmallWorld<float[], float> graph = GetGraph();
byte[] buffer = graph.SerializeGraph(); // buffer stores information about parameters and graph edges

// distance function must be the same as the one which was used for building the original graph
var copy = new SmallWorld<float[], float>(DotProductDistance.DotProductOptimized);
copy.DeserializeGraph(vectors, buffer); // the original vectors to attach to the "copy" vertices
```
##### Distance functions
The only one distance function supplied by the library is the cosine distance. But there are 4 versions to address universality/performance tradeoff.
```c#
CosineDistance.NonOptimized // most generic version works for all cases
CosineDistance.ForUnits     // gives correct result only when arguments are "unit" vectors
CosineDistance.SIMD         // uses SIMD instructions to optimize calculations
CosineDistance.SIMDForUnits // uses SIMD and requires arguments to be "units"
DotProductDistance.DotProductOptimized // Seperated, optimized with .NET 8 Tensor primitives for AVX (can fall back to non-hardware)
```
But the API allows to inject any custom distance function tailored specifically for your needs.

## Contributing
Your contributions and suggestions are very welcome! 

### How to contribute
If you've found a bug or have a feature request then please open an issue with detailed description.
We will be glad to see your pull requests as well.

1. Prepare workspace.
```
git clone https://github.com/bartczernicki/hnsw-sharp
cd HNSW.Net
git checkout -b [username]/[feature]
```
2. Update the library and add tests if needed.
3. Build and test the changes.
```
cd Src
dotnet build
dotnet test
```
4. Send the pull request from `[username]/[feature]` to `master` branch.
5. Get approve and merge the changes.

### Releasing
The library is distributed as a bundle of sources.

