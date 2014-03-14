image-bag
=========

Implements an image based bag of words classifier.

Key steps:

1. Feature points are extracted using SIFT.
2. Feature points are clustered using K Means to arrive at a set of cluster centroids
3. Training and testing images' feature points are quantized to those cluster centroids
4. Each image is now represented by a "TF" vector of cluster centroids
5. Using these vectors, a standard classifier like SVC or Random Forest Classifier can be trained.


For more information, refer to Professor Andrew Ng's Stanford [presentation](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&ved=0CDEQFjAB&url=http%3A%2F%2Fufldl.stanford.edu%2Feccv10-tutorial%2Feccv10_tutorial_part1.ppt&ei=dCYjU7LmIce-rgew7YGoDg&usg=AFQjCNEfYuybXqJhtGCE6balyOQyr3TOsA&cad=rja) on classical image classification methods.
