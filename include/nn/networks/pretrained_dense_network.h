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

 protected:

  void update_activation_trace_estimates();
  void update_utility_propagation_estimates();
  void update_dropout_utility_estimates(std::vector<float> inp, std::vector<float> normal_predictions, float dropout_perc);

 public:

  float perc_prune;
  int min_synapses_to_keep;
  int prune_interval;
  int start_pruning_at;
  int total_initial_synapses;
  float trace_decay_rate;
  std::vector<SyncedSynapse *> active_synapses;

  std::vector<std::vector<SyncedNeuron *>> all_neuron_layers;

  PretrainedDenseNetwork(int seed,
                         float perc_prune,
                         int min_synapses_to_keep,
                         int prune_interval,
                         int start_pruning_at,
                         float trace_decay_rate);

  ~PretrainedDenseNetwork();

  void load_relu_network(torch::jit::script::Module trained_model,
                         float step_size,
                         int no_of_input_features,
                         float utility_to_keep,
                         float trace_decay_rate);

  void load_linear_network(torch::jit::script::Module trained_model,
                           float step_size,
                           int no_of_input_features,
                           float utility_to_keep,
                           float trace_decay_rate);

  void print_graph(SyncedNeuron *root);
  void print_synapse_status();

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets);

  void update_weights();

  void prune_using_dropout_utility_estimator();
  void prune_using_utility_propoagation();
  void prune_using_trace_of_activation_magnitude();
  void prune_using_weight_magnitude_pruner();
  void prune_using_random_pruner();

  void prune_weights(std::string pruner);
  void update_utility_estimates(std::string pruner,
                                std::vector<float> input,
                                std::vector<float> prediction,
                                int dropout_iterations,
                                float dropout_perc);

  int get_current_synapse_schedule();
};


#endif //INCLUDE_NN_NETWORKS_DENSE_NETWORK_H_
