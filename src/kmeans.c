#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>

typedef struct dataset
{
  uint32_t len; // length of the dataset (number of rows)
  uint32_t attributes; // number of attributes (number of columns)
  float   **data;
} dataset_t;

// function prototypes
dataset_t load_dataset(char *filepath_ptr, char *delim);
void free_dataset(dataset_t *d);
void init_cv_subsets(dataset_t *src, dataset_t *testing, dataset_t *training,
                     uint32_t k_fold);
void load_cv_subsets(dataset_t *src, dataset_t *testing, dataset_t *training, 
                     uint32_t current_fold, uint32_t k_fold);
void free_subset(dataset_t *d);


int main (int argc, char *argv[])
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
  float   threshold = 1e-14; // allowed error between iterations for convergence
  int32_t max_iter = 100; // maximum allowed iterations in each k-means
  int32_t num_kmeans = 100; // number of parallel executed k-means
  int32_t k_fold = 10; // number of folds for cross validation
  int32_t seed = 0; // used to initialize the pseudo-random number generator
  int32_t num_threads = omp_get_num_threads();

  // process option flags
  uint32_t c = 0;
  opterr = 0;
  while ((c = getopt (argc, argv, "i:d:k:M:m:e:b:n:f:s:t:")) != -1)
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
    case 'e':
      if (atof(optarg) >= 0.0) {
        threshold = atof(optarg);
      } else {
        printf("Error: threshold for convergence must be a postive number.\n");
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
    case 's':
      if (atoi(optarg) >= 0) {
        seed = atoi(optarg);
      } else {
        printf("Error: seed must be a non-negative integer.\n");
        return 1;
      }
      break;
    case 't':
      if (atoi(optarg) <= 0) {
        printf("Error: number of threads must be a positve integer.\n");
        return 1;
      } else if (atoi(optarg) > omp_get_num_threads()) {
        printf("Error: maximum threads is %d.\n", omp_get_num_threads());
        return 1;
      } else {
        num_threads = atoi(optarg);
      }
      break;
    case '?':
      if (optopt == 'i' || optopt == 'd' || optopt == 'k' || optopt == 'M' || 
          optopt == 'm' || optopt == 'e' || optopt == 'b' || optopt == 'n' ||
          optopt == 'f' || optopt == 's' || optopt == 't' )
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

  dataset_t dataset = load_dataset(input_filepath_ptr, delimiter);
  // dataset_t testing = {};
  //   dataset_t training  = {};
  //   init_cv_subsets(&dataset, &testing, &training, k_fold);
  if (k == 0)
  { // user did not specify a value for k, so silhouette coefficients will be
    // used to select an optimal k between k_min and k_max.

    dataset_t testing = {};
    dataset_t training  = {};
    init_cv_subsets(&dataset, &testing, &training, k_fold);

    for (k = k_min; k <= k_max; ++k)
    {
      printf("Solving for k = %d\n", k);

      // k-fold cross validation
      for (int f = 0; f < k_fold; ++f)
      {

        load_cv_subsets(&dataset, &testing, &training, f, k_fold);
        printf("%3f %3f %3f\n", dataset.data[0][0], testing.data[0][0], training.data[0][0]);
        

        

      }

      

    }
    
    free_subset(&training);
    free_subset(&testing);
  }


  free_dataset(&dataset);

} // end main()




/* This function loads a dataset. Returns a dataset_t that must be freed by
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
  for (uint32_t i = 0; i < d->attributes; ++i)
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
  memcpy(training->data + (current_fold * fold_len), testing_start + fold_sz, (src->len * sizeof(float)) - fold_sz - testing_offset);

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
  //free(d->data);

  d->len = 0;
  d->attributes = 0;
  d->data = NULL;
  return;
} // end free_subset()


// TODO maybe?
/* Normalizes data points to values between 0 and 1.
 */
void normalize_dataset(dataset_t *d)
{
  uint32_t i, j;

  printf("Normalizing the data...\n");
  
  // find max for each attribute
  float max[d->attributes];
  memset(max, 0, d->attributes * sizeof(float));

  for(i = 0; i < d->len; i++) {
    for(j = 0; j < d->attributes; j++) {
      if(max[j] < d->data[i][j])
      {
        max[j] = d->data[i][j];
      }
    }
  }
  // Normalize the data by dividing each value by the max value of the column
  for(i = 0; i < d->len; i++) {
    for(j = 0; j < d->attributes; j++) {
        d->data[i][j] = d->data[i][j] / max[j];
    }
  }
} // end normalize_dataset()
