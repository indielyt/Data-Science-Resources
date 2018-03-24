import matplotlib.pyplot as plt



### DATACAMP INTRODUCTION TO UNSUPERVISED LEARNING

# Chapter 1: Clustering for dataset exploration
	# Clustering 2D points
	# Inspect clustering result
	# How many clusters of grain? (using inertia to determine K)
    # Evaluating the grain clustering
    # Scaling data for clusters (with a pipeline)
    # Cluster the fish data
    # Clustering stocks using KMeans
    # Which stocks move together (predicting correlation with clustering)

# Chapter 2: Visualization with hierarchical clustering and t-SNE

    # Heirarchical clustering of the grain data
    # Heirarchies of Stocks
    # Different Linkage, different heirarchical clustering
    # Extracting the cluster labels
    # t-SNE visualization of the grain dataset
    # A TSNE map of the stock market

# Chapter 3: Decorrelating your data and dimension reduction

    # Decorrelating the grain measurements with PCA
    # The first principle component
    # Variance of the PCA features
    # Dimension reduction of the fish measurements
    # A tf-idf word-frequency array
    # Clustering Wikipedia part I
    # Clustering Wikipedia part II

# Chapter 4: Discovering interpretable features
    # NMF applied to Wikipedia articles
    # NMF features of the Wikipedia articles
    # NMF learns topics of documents
    # Explore the LED digits dataset
    # Which articles are similar to 'Cristiano Ronaldo'? (recommender systems with cosine similarity)
    # Recommend musical artists part I (applying NMF)





#################### CHAPTER 1

### Clustering 2D points

# here we create a KMeans instance, and use it to fit and predict clusters
# from the points / new_points features.  The output is an array of values

# Import KMeans
from sklearn.cluster import KMeans 

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)





### Inspect clustering result

# Here we'll use the labels generated in previous exercise to plot the
# new_points clusters using labels for colors.  We'll also use the centroids
# attribute of the model instance from above to find centroids

# Import pyplot
import matplotlib.pyplot as plt

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()





### How many clusters of grain? (using inertia to determine K)

# Here we'll evaluate the decreasing inertia of the model by increasing
# K, the number of clusters.  A rule of thumb is to choose the K value 
# by a inertia graph where the elbow indicates reducing rate of improvement
# (see svg image in folder - 3 clusters were suggested as ideal for this data)
# Intertia is the distance of points from the cluster centroid

# Create range object for assessing # of clusters
ks = range(1, 6)

# Create empty list to track inertia
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(samples)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()






### Evaluating the grain clustering

# Cross tablulation counts the number of times each variety is associated with each
# cluster label.  Good fit is when a variety is clearly counted in one cluster
# more than the other (2) clusters.

# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

# Output:
varieties  Canadian wheat  Kama wheat  Rosa wheat
labels                                           
0                      68           9           0
1                       0           1          60
2                       2          60          10






### Scaling data for clusters (with a pipeline)

# Even clustering needs scaled data sets to work optimally,
# here we scale the dataset and run KMeans from within a pipeline
# StandardScaler removes the mean and scales unit variance to 1

# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)





### Cluster the fish data

# the pandas crosstab function is a nice way to evaluate 
# the results of clustering, notice the well separated clusters

# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)

# Calculate the cluster labels: labels
labels = pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'species': species, 'labels': labels})

# Create crosstab: ct
ct = pd.crosstab(df['labels'],df['species'])

# Display ct
print(ct)

# Output:
species  Bream  Pike  Roach  Smelt
labels                            
0           33     0      1      0
1            1     0     19      1
2            0    17      0      0
3            0     0      0     13





### Clustering stocks using KMeans

# Here we use normalizer, which transformes each feature
# of the stock 'movement' dataset independently of other features
# https://stackoverflow.com/questions/39120942/difference-between-standardscaler-and-normalizer-in-sklearn-preprocessing
# http://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html

# Import Normalizer
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer,kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

# Output:
Pipeline(steps=[('normalizer', Normalizer(copy=True, norm='l2')), ('kmeans', KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=10, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0))])




### Which stocks move together (predicting correlation with clustering)

# after clustering the stock price movements from above, we can
# use the labes inside a new dataframe to show us those clusters

# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))

# output
                             companies  labels
59                               Yahoo       0
2                               Amazon       0
17                     Google/Alphabet       0
35                            Navistar       1
58                               Xerox       1
7                                Canon       1
48                              Toyota       1
22                                  HP       1
21                               Honda       1
45                                Sony       1
15                                Ford       1
34                          Mitsubishi       1
32                                  3M       2
31                           McDonalds       2
41                       Philip Morris       2
23                                 IBM       2
20                          Home Depot       2
57                               Exxon       2
53                       Valero Energy       2
44                        Schlumberger       2
13                   DuPont de Nemours       2
12                             Chevron       2
10                      ConocoPhillips       2
8                          Caterpillar       2
28                           Coca Cola       3
38                               Pepsi       3
39                              Pfizer       4
43                                 SAP       4
46                      Sanofi-Aventis       4
6             British American Tobacco       4
30                          MasterCard       4
49                               Total       4
52                            Unilever       4
37                            Novartis       4
42                   Royal Dutch Shell       4
19                     GlaxoSmithKline       4
0                                Apple       5
33                           Microsoft       6
51                   Texas instruments       6
50  Taiwan Semiconductor Manufacturing       6
24                               Intel       6
47                            Symantec       6
11                               Cisco       6
14                                Dell       6
4                               Boeing       7
36                    Northrop Grumman       7
29                     Lookheed Martin       7
18                       Goldman Sachs       8
5                      Bank of America       8
26                      JPMorgan Chase       8
3                     American express       8
55                         Wells Fargo       8
1                                  AIG       8
16                   General Electrics       8
25                   Johnson & Johnson       9
40                      Procter Gamble       9
27                      Kimberly-Clark       9
54                            Walgreen       9
56                            Wal-Mart       9
9                    Colgate-Palmolive       9

























#################### CHAPTER 2










### Heirarchical clustering of the grain data

# See svg for output of dendrogram.  These visualizations start with
# every row as their own cluster, then merge the closets cluster, then 
# continue merging the closest clusters until only one cluster remains.
# Very useful for grouping and associations.  Looks like genetic tree

# Perform the necessary imports
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Calculate the linkage: mergings
mergings = linkage(samples, method='complete')

# Plot the dendrogram, using varieties as labels
dendrogram(mergings,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,)
plt.show()




### Heirarchies of Stocks

# Import normalize
from sklearn.preprocessing import normalize

# Normalize the movements: normalized_movements
normalized_movements = normalize(movements)

# Calculate the linkage: mergings
mergings = linkage(normalized_movements, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
plt.show()




### Different Linkage, different heirarchical clustering

# The linkage method can significantly affect the resulting clustering
# the 'complete' method calculates distance between the furthest apart samples
# of two clusters to determine which clusters to merge.  The 'single' method 
# calculates distance between the closest samples of two clusters

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

# Calculate the linkage: mergings
mergings = linkage(samples, method='single')

# Plot the dendrogram
dendrogram(mergings,labels=country_names, leaf_rotation=90, leaf_font_size=6)
plt.show()




### Extracting the cluster labels

# fcluster allow us to examine the results of any intermediate clustering
# we want to examine. The clustering is 'stopped' at the specified height
# which corresponds to distance between clusters.  The pd.crosstab function
# allows us to get the labels for those intermediate clusters. 

# Perform the necessary imports
import pandas as pd
from scipy.cluster.hierarchy import fcluster

# Use fcluster to extract labels: labels
labels = fcluster(mergings, 6,criterion='distance')

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

# Output:
varieties  Canadian wheat  Kama wheat  Rosa wheat
labels                                           
1                      14           3           0
2                       0           0          14
3                       0          11           0





### t-SNE visualization of the grain dataset

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=200)

# Apply fit_transform to samples: tsne_features
tsne_features = model.fit_transform(samples)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1st feature: ys
ys = tsne_features[:,1]

# Scatter plot, coloring by variety_numbers
plt.scatter(xs,ys,c=variety_numbers)
plt.show()




### A TSNE map of the stock market

# t-Distributed Stochastic Neighborhood Embedding is a visualization technique
# for dimensionality reduction that compressess multi-dimensional data onto a 
# 2d plane.  The axis are not of importance (or interpretibility), but the clustering
# is very useful to visually explore multidimensional data.  (see svg)

# Import TSNE
from sklearn.manifold import TSNE

# Create a TSNE instance: model
model = TSNE(learning_rate=50)

# Apply fit_transform to normalized_movements: tsne_features
tsne_features = model.fit_transform(normalized_movements)

# Select the 0th feature: xs
xs = tsne_features[:,0]

# Select the 1th feature: ys
ys = tsne_features[:,1]

# Scatter plot
plt.scatter(xs,ys,alpha=0.5)

# Annotate the points
for x, y, company in zip(xs, ys, companies):
    plt.annotate(company, (x, y), fontsize=5, alpha=0.75)
plt.show()


























#################### CHAPTER 3







### Decorrelating the grain measurements with PCA

# principle component analysis first decorrelates features 
# by shifting and aligning with the coordinate axes, effectively 'decorrelating'
# the relationship between the grain length and width.

# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)

# Output:
0.0






### The first principle component

# See svg plot of grains and first principle component represented as an arrow
# the first principle component is the direction in which the grain data
# varies the most

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()




### Variance of the PCA features

# See svg, creating a bar chart of variances by features (component).  Largest
# bar(s) indicates most variance - and principle component(s)

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()





### Dimension reduction of the fish measurements


# Here we define the number of components we want the dimension 
# reduction to stop at... in this case 2. 

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features = pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)

# Output:
(82,2)




### A tf-idf word-frequency array

documents = ['cats say meow', 'dogs say woof', 'dogs chase cats']

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)

# Output:
 [[ 0.51785612  0.          0.          0.68091856  0.51785612  0.        ]
     [ 0.          0.          0.51785612  0.          0.51785612  0.68091856]
     [ 0.51785612  0.68091856  0.51785612  0.          0.          0.        ]]
    ['cats', 'chase', 'dogs', 'meow', 'say', 'woof']






### Clustering Wikipedia part I

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components=50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters=6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)




### Clustering Wikipedia part II

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))

# Output:
                                          article  label
59                                    Adam Levine      0
50                                   Chad Kroeger      0
51                                     Nate Ruess      0
52                                     The Wanted      0
53                                   Stevie Nicks      0
58                                         Sepsis      0
55                                  Black Sabbath      0
56                                       Skrillex      0
57                          Red Hot Chili Peppers      0
54                                 Arctic Monkeys      0
19  2007 United Nations Climate Change Conference      1
18  2010 United Nations Climate Change Conference      1
16                                        350.org      1
15                                 Kyoto Protocol      1
17  Greenhouse gas emissions by the United States      1
13                               Connie Hedegaard      1
12                                   Nigel Lawson      1
11       Nationally Appropriate Mitigation Action      1
10                                 Global warming      1
14                                 Climate change      1
39                                  Franck Ribéry      2
38                                         Neymar      2
37                                       Football      2
36              2014 FIFA World Cup qualification      2
35                Colombia national football team      2
34                             Zlatan Ibrahimović      2
33                                 Radamel Falcao      2
32                                   Arsenal F.C.      2
31                              Cristiano Ronaldo      2
30                  France national football team      2
29                               Jennifer Aniston      3
27                                 Dakota Fanning      3
26                                     Mila Kunis      3
25                                  Russell Crowe      3
24                                   Jessica Biel      3
23                           Catherine Zeta-Jones      3
22                              Denzel Washington      3
21                             Michael Fassbender      3
20                                 Angelina Jolie      3
28                                  Anne Hathaway      3
49                                       Lymphoma      4
48                                     Gabapentin      4
47                                          Fever      4
46                                     Prednisone      4
45                                    Hepatitis C      4
43                                       Leukemia      4
42                                    Doxycycline      4
41                                    Hepatitis B      4
40                                    Tonsillitis      4
44                                           Gout      4
9                                        LinkedIn      5
8                                         Firefox      5
7                                   Social search      5
6                     Hypertext Transfer Protocol      5
5                                          Tumblr      5
4                                   Google Search      5
3                                     HTTP cookie      5
2                               Internet Explorer      5
1                                  Alexa Internet      5
0                                        HTTP 404      5

<script.py> output:
                                              article  label
    59                                    Adam Levine      0
    50                                   Chad Kroeger      0
    51                                     Nate Ruess      0
    52                                     The Wanted      0
    53                                   Stevie Nicks      0
    58                                         Sepsis      0
    55                                  Black Sabbath      0
    56                                       Skrillex      0
    57                          Red Hot Chili Peppers      0
    54                                 Arctic Monkeys      0
    19  2007 United Nations Climate Change Conference      1
    18  2010 United Nations Climate Change Conference      1
    16                                        350.org      1
    15                                 Kyoto Protocol      1
    17  Greenhouse gas emissions by the United States      1
    13                               Connie Hedegaard      1
    12                                   Nigel Lawson      1
    11       Nationally Appropriate Mitigation Action      1
    10                                 Global warming      1
    14                                 Climate change      1
    39                                  Franck Ribéry      2
    38                                         Neymar      2
    37                                       Football      2
    36              2014 FIFA World Cup qualification      2
    35                Colombia national football team      2
    34                             Zlatan Ibrahimović      2
    33                                 Radamel Falcao      2
    32                                   Arsenal F.C.      2
    31                              Cristiano Ronaldo      2
    30                  France national football team      2
    29                               Jennifer Aniston      3
    27                                 Dakota Fanning      3
    26                                     Mila Kunis      3
    25                                  Russell Crowe      3
    24                                   Jessica Biel      3
    23                           Catherine Zeta-Jones      3
    22                              Denzel Washington      3
    21                             Michael Fassbender      3
    20                                 Angelina Jolie      3
    28                                  Anne Hathaway      3
    49                                       Lymphoma      4
    48                                     Gabapentin      4
    47                                          Fever      4
    46                                     Prednisone      4
    45                                    Hepatitis C      4
    43                                       Leukemia      4
    42                                    Doxycycline      4
    41                                    Hepatitis B      4
    40                                    Tonsillitis      4
    44                                           Gout      4
    9                                        LinkedIn      5
    8                                         Firefox      5
    7                                   Social search      5
    6                     Hypertext Transfer Protocol      5
    5                                          Tumblr      5
    4                                   Google Search      5
    3                                     HTTP cookie      5
    2                               Internet Explorer      5
    1                                  Alexa Internet      5
    0                                        HTTP 404      5




























#################### CHAPTER 4










### NMF applied to Wikipedia articles

# Non Negative Matrix Factorization, see:
# https://www.slideshare.net/BenjaminBengfort/non-negative-matrix-factorization

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components=6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features)

# output: (note, output at this point is not interpretable)
[[  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   4.40544357e-01]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   5.66707580e-01]
     [  3.82045235e-03   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   3.98718448e-01]
     [  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
        0.00000000e+00   3.81808297e-01] .......





### NMF features of the Wikipedia articles

# Note that the output of NMF for these actors (reconstructed via non-negative 
# matrix factorization) show the 3rd NMF componnent wit the highest values.

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features,index=titles)

# Print the row for 'Anne Hathaway'
print(df.loc['Anne Hathaway'])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])

# output:
0    0.003845
1    0.000000
2    0.000000
3    0.575711
4    0.000000
5    0.000000
Name: Anne Hathaway, dtype: float64
0    0.000000
1    0.005601
2    0.000000
3    0.422380
4    0.000000
5    0.000000
Name: Denzel Washington, dtype: float64





### NMF learns topics of documents

# Here we use the NMF model of the Wikipedia articles to examine the most common
# components defining the topics of the documents. (Recall Anne Hathaway and Denzel
# Washington shared their highest NMF feature #3, which below we find is 
# the word'starred')

# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns=words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())

# Output:
(6, 13125)
film       0.627877
award      0.253131
starred    0.245284
role       0.211451
actress    0.186398
Name: 3, dtype: float64






### Explore the LED digits dataset

# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()

# Output:
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  1.  1.  0.  0.  0.  0.
  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.
  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.
  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
[[ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  1.  1.  1.  1.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  0.  0.  0.  0.  0.  0.  0.]]





 ### NMF learns the parts of images

 # here the NMF model learns the individual cells of an numerical LED screen
 # (think digital alarm clock number being made up of individual bars).  See svg

 def show_as_image(sample):
    bitmap = sample.reshape((13, 8))
    plt.figure()
    plt.imshow(bitmap, cmap='gray', interpolation='nearest')
    plt.colorbar()
    plt.show()

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Assign the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)




### PCA doesn't learn parts

# Unlike NMF, PCA doesn't learn the parts of things. Its components 
# do not correspond to topics (in the case of documents) or to parts of 
# images, when trained on images. See .svg

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components=7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)





### Which articles are similar to 'Cristiano Ronaldo'? (recommender systems with cosine similarity)

# .dot method performs cosine similarity (https://en.wikipedia.org/wiki/Cosine_similarity)
# For text matching, cosine similarity aims to reduce the difference between writing styles and
# superfulous text by examing the dot product angle and magnitude of word frequency vectors. 

# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())

# Output:
Cristiano Ronaldo                1.000000
Franck Ribéry                    0.999972
Radamel Falcao                   0.999942
Zlatan Ibrahimović               0.999942
France national football team    0.999923
dtype: float64





### Recommend musical artists part I (applying NMF)

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components=20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler,nmf,normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)





### Recommend musical artists part II

# here we'll use the pipeline to find artists similar to 
# Bruce Springsteen using NMF and cosine similarity

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features,index=artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print (similarities.nlargest())

# Output:
Bruce Springsteen    1.000000
Neil Young           0.958776
Leonard Cohen        0.918565
Van Morrison         0.884332
Bob Dylan            0.866569
dtype: float64
