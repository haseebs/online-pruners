#ifndef INCLUDE_NN_NETWORKS_DENSE_NETWORK_H_
#define INCLUDE_NN_NETWORKS_DENSE_NETWORK_H_


#include <vector>
#include <map>
#include <random>
#include <string>
#include "../synced_synapse.h"
#include "../synced_neuron.h"
#include "../dynamic_elem.h"
#include "./synced_network.h"
#include <torch/script.h>

class PretrainedDenseNetwork : public SyncedNetwork {

 public:

  float perc_prune;
  int min_synapses_to_keep;
  int prune_interval;
  int start_pruning_at;
  int total_initial_synapses;
  float trace_decay_rate;
  std::vector<SyncedSynapse *> active_synapses;

  std::vector<std::vector<SyncedNeuron *>> all_neuron_layers;

  PretrainedDenseNetwork(torch::jit::script::Module trained_model,
                         float step_size,
                         int seed,
                         int no_of_input_features,
                         float utility_to_keep,
                         float perc_prune,
                         int min_synapses_to_keep,
                         int prune_interval,
                         int start_pruning_at,
                         float trace_decay_rate);

  ~PretrainedDenseNetwork();

  void print_graph(SyncedNeuron *root);

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void imprint();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets);

  void update_weights();

  void add_feature(float step_size, float utility_to_keep);

  void add_feature_binary(float step_size, float utility_to_keep);

  void imprint_feature(int index, std::vector<float> feature, float step_size, float meta_step_size, int target);

  void imprint_feature_random(float step_size, float meta_step_size);

  void update_dropout_utility_estimates(std::vector<float> inp, std::vector<float> normal_predictions, float dropout_perc);
  void prune_using_dropout_utility_estimator();
  void prune_using_utility_propoagation();
  void prune_using_trace_of_activation_magnitude();
  void prune_using_weight_magnitude_pruner();
  void prune_using_random_pruner();

  void prune_weights(std::string pruner);

  int get_current_synapse_schedule();
};


#endif //INCLUDE_NN_NETWORKS_DENSE_NETWORK_H_
