#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "coco.h"

#include <iostream>
#include <nlshade/functions.h>
#include <nlshade/nlshade.h>

#define max(a, b) ((a) > (b) ? (a) : (b))

static const unsigned int BUDGET_MULTIPLIER = 10000;

static const long INDEPENDENT_RESTARTS = 1e5;

typedef void (*evaluate_function_t)(const double *x, double *y);

static coco_problem_t *PROBLEM;

static void evaluate_function(const double *x, double *y) {
  coco_evaluate_function(PROBLEM, x, y);
}

void example_experiment_nlshade(const char *suite_name,
                                const char *suite_options,
                                const char *observer_name,
                                const char *observer_options);

void my_random_search(evaluate_function_t evaluate_func,
                      evaluate_function_t evaluate_cons, const size_t dimension,
                      const size_t number_of_objectives,
                      const size_t number_of_constraints,
                      const double *lower_bounds, const double *upper_bounds,
                      const size_t number_of_integer_variables,
                      const size_t max_budget,
                      coco_random_state_t *random_generator);

typedef struct {
  size_t number_of_dimensions;
  size_t current_idx;
  char **output;
  size_t previous_dimension;
  size_t cumulative_evaluations;
  time_t start_time;
  time_t overall_start_time;
} timing_data_t;

static timing_data_t *timing_data_initialize(coco_suite_t *suite) {

  timing_data_t *timing_data =
      (timing_data_t *)coco_allocate_memory(sizeof(*timing_data));
  size_t function_idx, dimension_idx, instance_idx, i;

  /* Find out the number of all dimensions */
  coco_suite_decode_problem_index(suite,
                                  coco_suite_get_number_of_problems(suite) - 1,
                                  &function_idx, &dimension_idx, &instance_idx);
  timing_data->number_of_dimensions = dimension_idx + 1;
  timing_data->current_idx = 0;
  timing_data->output = (char **)coco_allocate_memory(
      timing_data->number_of_dimensions * sizeof(char *));
  for (i = 0; i < timing_data->number_of_dimensions; i++) {
    timing_data->output[i] = NULL;
  }
  timing_data->previous_dimension = 0;
  timing_data->cumulative_evaluations = 0;
  time(&timing_data->start_time);
  time(&timing_data->overall_start_time);

  return timing_data;
}

static void timing_data_time_problem(timing_data_t *timing_data,
                                     coco_problem_t *problem) {

  double elapsed_seconds = 0;

  if ((problem == NULL) || (timing_data->previous_dimension !=
                            coco_problem_get_dimension(problem))) {

    /* Output existing timing information */
    if (timing_data->cumulative_evaluations > 0) {
      time_t now;
      time(&now);
      elapsed_seconds = difftime(now, timing_data->start_time) /
                        (double)timing_data->cumulative_evaluations;
      timing_data->output[timing_data->current_idx++] =
          coco_strdupf("d=%lu done in %.2e seconds/evaluation\n",
                       timing_data->previous_dimension, elapsed_seconds);
    }

    if (problem != NULL) {
      /* Re-initialize the timing_data */
      timing_data->previous_dimension = coco_problem_get_dimension(problem);
      timing_data->cumulative_evaluations =
          coco_problem_get_evaluations(problem);
      time(&timing_data->start_time);
    }

  } else {
    timing_data->cumulative_evaluations +=
        coco_problem_get_evaluations(problem);
  }
}

static void timing_data_finalize(timing_data_t *timing_data) {

  /* Record the last problem */
  timing_data_time_problem(timing_data, NULL);

  if (timing_data) {
    size_t i;
    double elapsed_seconds;
    time_t now;
    int hours, minutes, seconds;

    time(&now);
    elapsed_seconds = difftime(now, timing_data->overall_start_time);

    printf("\n");
    for (i = 0; i < timing_data->number_of_dimensions; i++) {
      if (timing_data->output[i]) {
        printf("%s", timing_data->output[i]);
        coco_free_memory(timing_data->output[i]);
      }
    }
    hours = (int)elapsed_seconds / 3600;
    minutes = ((int)elapsed_seconds % 3600) / 60;
    seconds = (int)elapsed_seconds - (hours * 3600) - (minutes * 60);
    printf("Total elapsed time: %dh%02dm%02ds\n", hours, minutes, seconds);

    coco_free_memory(timing_data->output);
    coco_free_memory(timing_data);
  }
}

void example_experiment_nlshade(const char *suite_name,
                                const char *suite_options,
                                const char *observer_name,
                                const char *observer_options) {

  size_t run;
  coco_suite_t *suite;
  coco_observer_t *observer;
  timing_data_t *timing_data;

  suite = coco_suite(suite_name, "", suite_options);
  observer = coco_observer(observer_name, observer_options);
  timing_data = timing_data_initialize(suite);

  while ((PROBLEM = coco_suite_get_next_problem(suite, observer)) != NULL) {
    const unsigned dimension = coco_problem_get_dimension(PROBLEM);
    const auto *lower_bounds =
        coco_problem_get_smallest_values_of_interest(PROBLEM);
    const auto *upper_bounds =
        coco_problem_get_largest_values_of_interest(PROBLEM);

    auto evaluations_done =
        static_cast<long>(coco_problem_get_evaluations(PROBLEM));

    auto evaluations_remaining =
        static_cast<long>((dimension * BUDGET_MULTIPLIER) - evaluations_done);

    auto eval_func = [&](const auto *input, const auto size) -> double {
      double result;
      evaluate_function(input, &result);
      return result;
    };

    const unsigned population_size = 30 * dimension;
    const int memory_size = 20 * dimension;
    const double archive_size = 2.1;

    nlshade::Optimizer optim{
      eval_func,
      population_size,
      dimension,
      memory_size,
      archive_size,
      static_cast<int>(evaluations_remaining),
      lower_bounds[0],
      upper_bounds[0]
    };
    std::ignore = optim.optimize();

   /* if ((coco_problem_final_target_hit(PROBLEM) &&*/
         /*coco_problem_get_number_of_constraints(PROBLEM) == 0) ||*/
        /*(evaluations_remaining <= 0)) {*/
      /*break;*/
    /*}*/

    if (coco_problem_get_evaluations(PROBLEM) == evaluations_done) {
      printf("WARNING: Budget has not been exhausted (%lu/%lu evaluations "
             "done)!\n",
             (unsigned long)evaluations_done,
             (unsigned long)dimension * BUDGET_MULTIPLIER);
      break;
    } else if (coco_problem_get_evaluations(PROBLEM) < evaluations_done)
      coco_error("Something unexpected happened - function evaluations were "
                 "decreased!");
    timing_data_time_problem(timing_data, PROBLEM);
  }
  timing_data_finalize(timing_data);

  coco_observer_free(observer);
  coco_suite_free(suite);
}

int main(void) {
  coco_set_log_level("info");

  printf("Running the example NLSHADE experiment... (might take time, be "
         "patient)\n");
  fflush(stdout);

  example_experiment_nlshade("bbob", "", "bbob",
                             "result_folder: NLSHADE_on_bbob");

  printf("Done!\n");
  fflush(stdout);

  return 0;
}
