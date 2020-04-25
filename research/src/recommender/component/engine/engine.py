from recommender.component.embedding.embedder import RandomEmbedder, FastTextEmbedder

print(FastTextEmbedder('C:/x/diplomka/research/model/fasttext/cc.cs.300.bin').embed("ahoj"))