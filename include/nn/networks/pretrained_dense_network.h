#ifndef INCLUDE_NN_NETWORKS_PRETRAINED_DENSE_H_
#define INCLUDE_NN_NETWORKS_PRETRAINED_DENSE_H_


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

  std::vector<SyncedSynapse *> active_synapses;

  std::vector<std::vector<SyncedNeuron *>> all_neuron_layers;

  PretrainedDenseNetwork(torch::jit::script::Module trained_model,
                         float step_size,
                         int seed,
                         int no_of_input_features,
                         int total_targets,
                         float utility_to_keep);

  ~PretrainedDenseNetwork();

  void print_graph(SyncedNeuron *root);

  void viz_graph();

  void set_print_bool();

  std::string get_viz_graph();

  void imprint();

  void forward(std::vector<float> inputs);

  void backward(std::vector<float> targets, bool update_val);

  void add_feature(float step_size, float utility_to_keep);

  void add_feature_binary(float step_size, float utility_to_keep);

  void imprint_feature(int index, std::vector<float> feature, float step_size, float meta_step_size, int target);

  void imprint_feature_random(float step_size, float meta_step_size);
};


#endif //INCLUDE_NN_NETWORKS_PRETRAINED_DENSE_NETWORK
