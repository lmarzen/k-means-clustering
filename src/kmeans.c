#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include <unistd.h>

/* Structure for storing 2d arrays of data
 */
typedef struct dataset
{
  uint32_t len; // length of the dataset (number of rows)
  uint32_t attributes; // number of attributes (number of columns)
  float    **data;
} dataset_t;

/* Structure for storing 1d arrays of data that indicate what cluster each
 * datapoint belongs to.
 */
typedef struct labels
{
  uint32_t len; // length of the dataset (number of rows)
  uint32_t *data;
} labels_t;

/* Structure for storing results from kmeans.
 */
typedef struct results
{
  dataset_t centroids;
  labels_t  labels;
  float     sse; // sum of squared estimate of errors (SSE)
} results_t;



// function prototypes
dataset_t load_dataset(char *filepath_ptr, char *delim);
void free_dataset(dataset_t *d);
void init_cv_subsets(dataset_t *src, dataset_t *testing, dataset_t *training,
                     uint32_t k_fold);
void load_cv_subsets(dataset_t *src, dataset_t *testing, dataset_t *training, 
                     uint32_t current_fold, uint32_t k_fold);
void free_subset(dataset_t *d);
float silhouette_analysis(dataset_t *testing, dataset_t *training, 
                      uint32_t num_kmeans, uint32_t k, uint32_t max_iter, 
                      uint32_t print);
results_t kmeans(dataset_t *d, uint32_t k, uint32_t max_iter);
void free_results(results_t *d);
void normalize(dataset_t *d);
dataset_t init_centroids(uint32_t k, uint32_t num_attributes);
void rand_centroids(dataset_t *src, dataset_t *centroids);
float sq_euclidean_dist(float *a, float *b, int dim);
float euclidean_dist(float *a, float *b, int dim);
void assign_nearest_cluster(dataset_t *d, labels_t *data_labels, 
                            dataset_t *centroids);
uint32_t update_centroid_position(dataset_t *d, labels_t *data_labels, 
                                  dataset_t *centroids);
float calc_sse(dataset_t *d, labels_t *data_labels, dataset_t *centroids);
float calc_silhouette(dataset_t *testing, labels_t *test_labels, 
                      dataset_t *centroids);
void print_dataset(dataset_t *d, char *delim);
void print_dataset_pretty(dataset_t *d);


int main(int argc, char *argv[])
{
  // default options
  char    *input_filepath_ptr = "input.csv";
  char    *output_clusters_filepath_ptr = "output_clusters.csv";
  char    *output_centroids_filepath_ptr = "output_centroids.csv";
  char    *delimiter = ","; // used when reading input and when write output
  int32_t k = 0; // k == 0 will indicate that silhouette coefficients should
                 // determine optimal k from k_min to k_max
  int32_t k_min = 2;
  int32_t k_max = 10;
  int32_t max_iter = 100; // maximum allowed iterations in each k-means
  int32_t num_kmeans = 100; // number of parallel executed k-means
  int32_t k_fold = 10; // number of folds for cross validation

  // process option flags
  uint32_t c = 0;
  opterr = 0;
  while ((c = getopt (argc, argv, "i:d:k:M:m:b:n:f:t:")) != -1)
  {
    switch (c) 
    {
    case 'i':
      input_filepath_ptr = optarg;
      break;
    case 'd':
      delimiter = optarg;
      break;
    case 'k':
      if (atoi(optarg) > 0) {
        k = atoi(optarg);
      } else {
        printf("Error: k must be a postive integer.\n");
        return 1;
      }
      break;
    case 'M':
      if (atoi(optarg) > 0) {
        k_max = atoi(optarg);
      } else {
        printf("Error: k_max must be a postive integer.\n");
        return 1;
      }
      break;
    case 'm':
      if (atoi(optarg) > 0) {
        k_min = atoi(optarg);
      } else {
        printf("Error: k_min must be a postive integer.\n");
        return 1;
      }
      break;
    case 'b':
      if (atoi(optarg) > 0) {
        max_iter = atoi(optarg);
      } else {
        printf("Error: maximum iterations must be a postive integer.\n");
        return 1;
      }
      break;
    case 'n':
      if (atoi(optarg) > 0) {
        k_min = atoi(optarg);
      } else {
        printf("Error: number of parallel k-means must be a postive integer.\n");
        return 1;
      }
      break;
    case 'f':
      if (atoi(optarg) > 0) {
        k_fold = atoi(optarg);
      } else {
        printf("Error: k-folds must be a postive integer.\n");
        return 1;
      }
      break;
    case 't':
      if (atoi(optarg) <= 0) {
        printf("Error: number of threads must be a positve integer.\n");
        return 1;
      // } else if (atoi(optarg) > omp_get_num_threads()) {
      //   printf("Error: maximum threads is %d.\n", omp_get_num_threads());
      //   return 1;
      } else {
        omp_set_num_threads(atoi(optarg));
      }
      break;
    case '?':
      if (optopt == 'i' || optopt == 'd' || optopt == 'k' || optopt == 'M' || 
          optopt == 'm' || optopt == 'e' || optopt == 'b' || optopt == 'n' ||
          optopt == 'f' || optopt == 't' )
        fprintf (stderr, "Option -%c requires an argument.\n", optopt);
      else if (isprint (optopt))
        fprintf (stderr, "Unknown option '-%c'.\n", optopt);
      else
        fprintf (stderr, "Unknown option character '\\x%x'.\n", optopt);
      return 1;
    default:
      return 1;
    }
  } // end options while-loop

  // ignore invalid options, but alert user.
  for (uint32_t i = optind; i < argc; i++)
  {
    printf ("Non-option argument %s\n", argv[i]);
  }

  // check for valid k_max and k_min
  if (k_max <= k_min)
  {
    printf("Error: k_max must be greater than k_min.\n");
    return 1;
  }

  // generate random seed
  struct timeval time;
  gettimeofday(&time, NULL);
  uint32_t seed = (time.tv_usec ^ (time.tv_sec << 20)) & 0xffffffff;
  srand(seed);

  dataset_t dataset = load_dataset(input_filepath_ptr, delimiter);
  normalize(&dataset);

  if (k == 0)
  { // user did not specify a value for k, so silhouette coefficients will be
    // used to select an optimal k between k_min and k_max.

    dataset_t testing = {};
    dataset_t training  = {};
    init_cv_subsets(&dataset, &testing, &training, k_fold);

    for (k = k_min; k <= k_max; ++k)
    {
      printf("Performing silhouette analysis for k = %d\n", k);

      // k-fold cross validation
      for (int f = 0; f < k_fold; ++f)
      {
        load_cv_subsets(&dataset, &testing, &training, f, k_fold);
        // TODO
        // ...
      }
    }

    free_subset(&training);
    free_subset(&testing);
  }

  float sil = silhouette_analysis(&dataset, &dataset, num_kmeans, k, max_iter, 1);
  printf("sil: %f", sil);
  
  

  // results_t results = kmeans(&dataset, 3, max_iter);
  // free_results(&results);

  // TODO print to file


  free_dataset(&dataset);
} // end main()


/* Executes kmeans for num_kmeans times.
 * 
 * Returns the silhouette coeffient for the best clusters.
 */
float silhouette_analysis(dataset_t *testing, dataset_t *training, 
                          uint32_t num_kmeans, uint32_t k, uint32_t max_iter, 
                          uint32_t print)
{
  results_t best_results, tmp_results;

  best_results = kmeans(training, k, max_iter);;

  #pragma omp parallel for private(tmp_results)
  for (uint32_t i = 1; i < num_kmeans; ++i)
  {
    tmp_results = kmeans(training, k, max_iter);
    
    #pragma omp critical
    if (tmp_results.sse < best_results.sse)
    {
      free_results(&best_results);
      best_results = tmp_results;
    }
    else
    {
      free_results(&tmp_results);
    }
  }

  // evaluate results agains testing set
  labels_t test_data_labels;
  test_data_labels.data = (uint32_t *) malloc(testing->len * sizeof(uint32_t));
  test_data_labels.len = testing->len;
  assign_nearest_cluster(testing, &test_data_labels, &best_results.centroids);
  float sil = calc_silhouette(testing, &test_data_labels, 
                              &best_results.centroids);
  free(test_data_labels.data);



  if (print)
  {
    // print best clusters to terminal
    // write output files
  }
  free_results(&best_results);
  return sil;
} // end silhouette_analysis()


/* Executes the kmeans algorithm until convergence or max_iter.
 * 
 * Returns a results_t containing of centroids, data labels(what cluster each
 * data point belongs to) and the sum of squared estimate of errors (SSE).
 */
results_t kmeans(dataset_t *d, uint32_t k, uint32_t max_iter)
{
  results_t r = {};
  r.centroids = init_centroids(k, d->attributes);
  rand_centroids(d, &r.centroids); // centroids are assigned to random 
                                         // data points

  // array to store the index of the nearest cluster(aka centroid) for each 
  // data point
  r.labels.data = (uint32_t *) malloc(d->len * sizeof(uint32_t));
  r.labels.len = d->len;

  uint32_t has_converged = 0;
  uint32_t i = 0;
  do
  {
    // assign label to each datapoint to indicate nearest cluster
    assign_nearest_cluster(d, &r.labels, &r.centroids);
    // update the position of each centoid based on datapoint labels
    has_converged = update_centroid_position(d, &r.labels, &r.centroids);
    ++i;
  } while ( !has_converged && (i < max_iter) );

  r.sse = calc_sse(d, &r.labels, &r.centroids);
  
  return r;
} // end kmeans()


/* Releases all resources stored used by results_t.
 */
void free_results(results_t *r)
{
  free_dataset(&r->centroids);
  free(r->labels.data);
  r->labels.len = 0;
  r->sse = 0;
  return;
} // end free_results()


/* This function parses a data file. Returns a dataset_t that must be freed by
 * calling free_dataset(...).
 *
 * Assumes data begins on first line and consists of attributes represented by 
 * floats and seperated by the specified delimiter (delim).
 */
dataset_t load_dataset(char *filepath_ptr, char *delim)
{
    FILE *fp = NULL;
    dataset_t d = {};

    fp = fopen(filepath_ptr, "r");

    if (fp == NULL)
    {
      printf("Error: Unable to open file : %s\n", filepath_ptr);
      exit(1);
    }
    else 
    {
      printf("Loading dataset from file: %s\n", filepath_ptr);
    }

    char buf[1024] = {};

    // count the number of attributes in the first line
    if (fgets(buf, sizeof(buf), fp)) {
      strtok(buf, delim);
      d.attributes = 1;
          
      // after every delimiter we know there should be another attribute
      while (strtok(NULL, delim)) {
        ++d.attributes;
      }
    }
    rewind(fp); // move fp back to beginning of the file
    printf("attributes: %d\n", d.attributes);
    
    uint32_t rows = 128; // start by allocating 128 rows of space
    d.data = (float **) malloc(rows * sizeof(float *));

    while (fgets(buf, sizeof(buf), fp)) {
        d.data[d.len] = (float *) calloc(d.attributes, sizeof(float));
        char *token = strtok(buf, delim);
        uint32_t i = 0;
        while (token != NULL) {
            d.data[d.len][i] = atof(token);
            token = strtok(NULL, delim);
            ++i;
        }

        ++d.len;
        // if we fill up all the allocated space, double it
        if(d.len > rows) {
            rows = rows * 2;
            d.data = (float **) realloc(d.data, rows * sizeof(float *));
        }
    }
    printf("length: %d\n", d.len);

    fclose(fp);

    return d;
} // end load_dataset()


/* Frees the dynamically allocated data in dataset_t.
 */
void free_dataset(dataset_t *d)
{
  for (uint32_t i = 0; i < d->len; ++i)
  {
    free(d->data[i]);
  }
  free(d->data);

  d->len = 0;
  d->attributes = 0;
  d->data = NULL;
  return;
} // end free_dataset()


/* Allocates the space for the testing and training data subsets. Used for 
 * k-fold cross-validation. Free by calling free_subset(...).
 */
void init_cv_subsets(dataset_t *src, dataset_t *testing, dataset_t *training, 
                     uint32_t k_fold)
{
  uint32_t fold_len = src->len / k_fold;
  size_t fold_sz = fold_len * sizeof(float *);

  testing->attributes  = src->attributes;
  training->attributes = src->attributes;
  testing->len  = fold_len;
  training->len = src->len - fold_len;
  testing->data  = (float **) malloc( fold_sz );
  training->data = (float **) malloc( training->len * sizeof(float *) );

  return;
} // end init_cv_subsets()


/* Copies pointers from source dataset, d, into testing and training subsets for
 * the current fold iteration. Used for k-fold cross-validation.
 */
void load_cv_subsets(dataset_t *src, dataset_t *testing, dataset_t *training, 
                     uint32_t current_fold, uint32_t k_fold)
{
  uint32_t fold_len = src->len / k_fold;
  size_t fold_sz = fold_len * sizeof(float *);
  size_t testing_offset = current_fold * fold_sz;
  float **testing_start = src->data + (current_fold * fold_len);
  // The memcpy's here will copy the pointers from the source dataset into the
  // subsets
  //
  // ex:
  //   source dataset: XXXOX
  //   current_fold = 3
  //   k_fold = 5
  //   X = training partition
  //   0 = testing partition
  //
  // copy first part of training data: XXX
  memcpy(training->data, src->data, testing_offset);
  // copy all of testing data: O
  memcpy(testing->data, testing_start, fold_sz);
  // copy the remaining data in source to the remaining space in training: X
  memcpy(training->data + (current_fold * fold_len), 
         testing_start + testing->len, 
         (src->len * sizeof(float *)) - fold_sz - testing_offset);

  return;
} // end load_cv_subsets()


/* Frees the dynamically allocated data in dataset_t, specifically for subsets
 * to avoid double free.
 */
void free_subset(dataset_t *d)
{
  // only a single array of pointers is allocated for the purpose of the
  // subsets, the real data is in the source dataset and should be freed by
  // calling free_dataset
  free(d->data);

  d->len = 0;
  d->attributes = 0;
  d->data = NULL;
  return;
} // end free_subset()


/* Normalizes data points to values between 0 and 1.
 */
void normalize(dataset_t *d)
{
  uint32_t i, j;

  printf("Normalizing the dataset...\n");
  
  float max[d->attributes];
  memset(max, 0, d->attributes * sizeof(float));
  // find max for each attribute
  for (i = 0; i < d->len; i++)
  {
    for (j = 0; j < d->attributes; j++)
    {
      if (d->data[i][j] > max[j])
      {
        max[j] = d->data[i][j];
      }
    }
  }

  float min[d->attributes];
  memcpy(min, max, d->attributes * sizeof(float));
  // find min for each attribute
  for (i = 0; i < d->len; i++)
  {
    for (j = 0; j < d->attributes; j++)
    {
      if (d->data[i][j] < min[j])
      {
        min[j] = d->data[i][j];
      }
    }
  }

  // calculate the range of each attribute
  float range[d->attributes];
  for (i = 0; i < d->attributes; i++)
  {
    range[i] = max[i] - min[i];
  }

  // Normalize the data... xnormalized = (x - xminimum) / range of x
  for (i = 0; i < d->len; i++)
  {
    for (j = 0; j < d->attributes; j++)
    {
      d->data[i][j] = (d->data[i][j] - min[j]) / range[j];
    }
  }

  return;
} // end normalize()


/* Allocates space for k centroids with the specified number of attributes.
 * Free centroids by calling free_dataset(...).
 */
dataset_t init_centroids(uint32_t k, uint32_t num_attributes)
{
  dataset_t centroids = {};
  centroids.len = k;
  centroids.attributes = num_attributes;
  centroids.data = (float **) malloc(k * sizeof(float *));
  for (uint32_t i = 0; i < k; ++i)
  {
    centroids.data[i] = (float *) malloc(num_attributes * sizeof(float));
  }
  return centroids;
} // end init_centroids()


/* Selects random points and copys them to centroids.
 * Assumes that srand() has been called and that centroids has been initialized
 * by a previous call to init_centroids.
 */
void rand_centroids(dataset_t *src, dataset_t *centroids)
{
  // points must be selected without replacement, so we must keep track of 
  // selected points
  uint32_t *rand_indices = malloc(centroids->len * sizeof(uint32_t));

  for (uint32_t i = 0; i < centroids->len; ++i)
  {
    uint32_t valid_index = 0;
    while (!valid_index)
    {
      rand_indices[i] = rand() % src->len;
      valid_index = 1;
      // check if index has been used before, if it has index is not valid
      for (uint32_t j = 0; j < i; ++j)
      {
        if (rand_indices[i] == rand_indices[j])
        {
          valid_index = 0;
          break;
        }
      }
    }

    memcpy(centroids->data[i], 
           src->data[rand_indices[i]], 
           centroids->attributes * sizeof(float));
  }

  free(rand_indices);
} // end rand_centroids()


/* Returns the (squared) Euclidean distance between two points, a and b, of 
 * arbitrary dimension, dim. 
 *
 * NOTE: Answer is left squared because kmeans only cares about relative 
 *       distance so it is a waste of resources to compute the sqrt.
 */
float sq_euclidean_dist(float *a, float *b, int dim)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < dim; i++)
  {
    sum += pow(a[i] - b[i], 2);
  }
  return sum; // sqrt(sum) would be the true Euclidean distance
} // end sq_euclidean_dist()

/* Returns the Euclidean distance between two points, a and b, of 
 * arbitrary dimension, dim.
 */
float euclidean_dist(float *a, float *b, int dim)
{
  return sqrt(sq_euclidean_dist(a, b, dim));
} // end euclidean_dist()


/* Assign label to each datapoint to indicate nearest cluster.
 */
void assign_nearest_cluster(dataset_t *d, labels_t *data_labels, 
                            dataset_t *centroids)
{
  // start by assigning all data to the first(index 0) cluster
  memset(data_labels->data, 0, d->len * sizeof(uint32_t));

  for (uint32_t i = 0; i < d->len; ++i)
  {
    float min_dist = sq_euclidean_dist(d->data[i], centroids->data[0], 
                                       d->attributes);
    for (uint32_t j = 1; j < centroids->len; ++j)
    {
      float tmp_dist = sq_euclidean_dist(d->data[i], centroids->data[j], 
                                         d->attributes);
      if (tmp_dist < min_dist)
      {
        min_dist = tmp_dist;
        data_labels->data[i] = j;
      }
    }
  }
} // end assign_nearest_cluster()


/* Update the position of each centoid based on datapoint labels.
 * If a centroid is empty (no points belong to it) it will be re-initialized.
 *
 * Returns a bool to indicate convergence.
 *   0 centroids have not converged. (centroid positions did change)
 *   1 centroids have converged. (centroid positions did not change)
 */
uint32_t update_centroid_position(dataset_t *d, labels_t *data_labels, 
                                  dataset_t *centroids)
{
  // allocate space for new centroids
  float **new_centroids = (float **) malloc(centroids->len * sizeof(float *));
  for (uint32_t i = 0; i < centroids->len; ++i)
  { // it is important that new_centroids data is initialized to 0
   new_centroids[i] = (float *) calloc(centroids->attributes, sizeof(float));
  }

  // sum the vectors and record the number of points that belong to each cluster
  uint32_t *record = (uint32_t *) calloc(centroids->len, sizeof(uint32_t));
  for (uint32_t i = 0; i < d->len; ++i)
  {
    for (uint32_t j = 0; j < d->attributes; ++j)
    {
      new_centroids[data_labels->data[i]][j] += d->data[i][j];
    }
    record[data_labels->data[i]] += 1;
  }

  // finish computing the average position of each cluster
  for (uint32_t i = 0; i < centroids->len; ++i)
  {
    if (record[i] > 0)
    {
      for (uint32_t j = 0; j < centroids->attributes; ++j)
      {
        new_centroids[i][j] = new_centroids[i][j] / ((float) record[i]);
      }
    }
    else
    { // centroid has no points that belong to it... reinitialize centroid to
      // a random datapoint
      memcpy(new_centroids[i], 
             d->data[rand() % d->len], 
             centroids->attributes * sizeof(float));
    }
  }
  

  // compare old and new centroid values to determine if data has converged
  uint32_t has_converged = 1;
  for (uint32_t i = 0; i < centroids->len; ++i)
  {
    for (uint32_t j = 0; j < centroids->attributes; ++j)
    {
      if (centroids->data[i][j] != new_centroids[i][j])
      {
        has_converged = 0;
        break;
      }
    }
  }

  free(centroids->data);
  centroids->data = new_centroids;

  free(record);

  return has_converged;
} // end update_centroid_position()


/* Calculates the sum of squared estimate of errors (SSE) between centroids and
 * their points.
 */
float calc_sse(dataset_t *d, labels_t *data_labels, dataset_t *centroids)
{
  float sum = 0.0;
  for (uint32_t i = 0; i < d->len; i++)
  {
    sum += sq_euclidean_dist(d->data[i], centroids->data[data_labels->data[i]], 
                             d->attributes);
  }
  return sum;
} // end calc_sse()


/* Calculates and returns the silhouette coeffient.
 *
 * s_i = (b_i - a_i) / max(a_i, b_i)
 * 
 * s_i : is the silhouette coeffient for a datapoint
 * a_i : is the intra cluster distance defined as the average distance from all 
 *       data points in the same cluster 
 * b_i : is the inter cluster distance defined as the average distance from all 
 *       data points in the closest cluster
 * 
 * The coefficient can take values in the interval [-1, 1].
 *   0 : the sample is very close to the neighboring clusters
 *   1 : the sample is far away from the neighboring clusters
 *  -1 : the sample is assigned to the wrong clusters
 */
float calc_silhouette(dataset_t *testing, labels_t *test_labels, 
                      dataset_t *centroids)
{
  float mean_sil = 0.0;

  #pragma omp parallel for
  for (uint32_t i = 0; i < testing->len; ++i)
  {
    uint32_t C = 0; // number of datapoints in the same cluster
    float dist = 0.0;
    for (uint32_t j = 0; j < test_labels->len; ++j)
    {
      if ( (i != j) && (test_labels->data[i] == test_labels->data[j]) )
      {
        dist += euclidean_dist(testing->data[i], testing->data[j], 
                               testing->attributes);
        C += 1;
      }
    }
    float a_i = dist / ((float) (C));

    float b_i = -1;
    for (uint32_t k = 0; k < centroids->len; ++k)
    { // find closest cluster (that is not its own)
      float tmp_dist = euclidean_dist(centroids->data[k], testing->data[i], 
                                      centroids->attributes);
      if ((tmp_dist < b_i || b_i < 0.0) && k != test_labels->data[i])
      {
        b_i = tmp_dist;
      }
    }


    float s_i = (b_i - a_i) / fmax(a_i, b_i);
    mean_sil += s_i;
  }

  mean_sil = mean_sil / ((float) testing->len);

  return mean_sil;
} // calc_silhouette()

/* Prints a dataset with the specified delimiter.
 */
void print_dataset(dataset_t *d, char *delim)
{
  for (uint32_t i = 0; i < d->len; ++i)
  {
    uint32_t j;
    for (j = 0; j < d->attributes - 1; ++j)
    {
      printf("%f%s", d->data[i][j], delim);
    }
    printf("%f\n", d->data[i][j]);
  }
  printf("\n");
} // end print_dataset()

/* Prints a dataset in a pretty to read format.
 */
void print_dataset_pretty(dataset_t *d)
{
  for (uint32_t i = 0; i < d->len; ++i)
  {
    uint32_t j;
    putchar('{');
    for (j = 0; j < d->attributes - 1; ++j)
    {
      printf("%f, ", d->data[i][j]);
    }
    printf("%f}\n", d->data[i][j]);
  }
  printf("\n");
} // end print_dataset_pretty()
