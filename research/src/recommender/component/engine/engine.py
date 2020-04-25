from recommender.component.embedding import RandomEmbedder, FastTextEmbedder

print(RandomEmbedder(300).embed("ahoj"))