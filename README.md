# k-means-clustering
This project is a C implementation of the k-means clustering algorithm that has been parallelized to run across multiple threads with OpenMP and uses silhouette coefficients to find an optimal number of clusters.

This algorithm first attempts to identify an optimal number of clusters to solve for, using silhouette coefficients that are averaged over k-folds. The dataset is parsed from a file and split into training and testing datasets and uses k-folds cross-validation. Once silhouette coefficients have been calculated for a range of k values, a target k is selected, and centroids are calculated on the entire dataset.

This implementation can handle datasets of arbitrary dimension and length. The expected input format is comma-separated, but the delimiter can be changed with the '-d' flag. For an example dataset, see data/iris.csv.

Two output files will be generated in the directory of the binary. The first 'output_clusters.csv' will be the dataset with an additional column indicating which cluster each point belongs to. The second file is 'output_centroids.csv', which contains the coordinates of the centroids.


Options
---
-i [filepath]
<ul>
Input filename/path.
</ul><ul>
Default is 'input.cvs'
</ul>
-d [delimiter]
<ul>
Delimiter used when parsing the input dataset file.
</ul><ul>
Default delimiter is ",".
</ul>
-k [num_clusters]
<ul>
Specify the number of clusters to identify, k. If you know the number of clusters that should be identified, you can pass this option to bypass using silhouette analysis.
</ul><ul>
Must be a positive integer.
</ul>
-m [min]
<ul>
Specify the minimum number of clusters to analyze during silhouette analysis.
</ul><ul>
Must be a positive integer.
</ul><ul>
Default is 2.
</ul>
-M [max]
<ul>
Specify the maximum number of clusters to analyze during silhouette analysis.
</ul><ul>
Must be a positive integer.
</ul><ul>
Default is 10.
</ul>
-b [max_iterations]
<ul>
Maximum allowed iterations in each k-means.
</ul><ul>
Must be a positive integer.
</ul><ul>
Default is 100.
</ul>
-e [num_kmeans]
<ul>
Number of parallel executed k-means.
</ul><ul>
Must be a positive integer.
</ul><ul>
Default is 100.
</ul>
-f [num_folds]
<ul>
Number of folds for cross-validation.
</ul><ul>
Must be a positive integer.
</ul><ul>
Default is 5.
</ul>
-t [num_threads]
<ul>
Number of threads to spread the workload across.
</ul><ul>
Must be a positive integer.
</ul><ul>
Default behavior will use all available threads.
</ul>
-r
<ul>
Randomize the dataset order. It is important that the dataset is randomized for cross-validation.
</ul>
-n
<ul>
Normalize the dataset. This is a good idea if the dataset is not already normalized.
</ul>

Getting Started
---

Linux:

* Clone repository `git clone https://github.com/lmarzen/k-means-clustering.git` or download and extract ZIP.

* Open a terminal(or command prompt on Windows) in the src directory and run `make` to build the program.

* Run the program by typing `./kmeans` followed by any valid arguments.

* Done.