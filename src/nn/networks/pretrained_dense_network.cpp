#include <execution>
#include <algorithm>
#include <iostream>
#include "../../../include/nn/networks/pretrained_dense_network.h"

#include <torch/script.h>

using namespace torch::indexing;

PretrainedDenseNetwork::PretrainedDenseNetwork(torch::jit::script::Module trained_model,
                                               float step_size,
                                               int seed,
                                               int no_of_input_features,
                                               int total_targets,
                                               float utility_to_keep) {

	this->mt.seed(seed);

	for (int i = 0; i < no_of_input_features; i++) {
		SyncedNeuron *n = new LinearSyncedNeuron(true, false);
		n->neuron_age = 10000000;
		n->drinking_age = 0;
		n->set_layer_number(0);
		this->input_neurons.push_back(n);
		this->all_neurons.push_back(n);
	}

	int current_layer_number = 1;
	for (const auto& param_group : trained_model.parameters()) {
		std::vector<SyncedNeuron*> curr_layer;

		for (int neuron_idx = 0; neuron_idx < param_group.size(0); neuron_idx++) {
			SyncedNeuron *n;
			if (current_layer_number == trained_model.parameters().size()) {
				n = new SigmoidSyncedNeuron(false, true);
				this->output_neurons.push_back(n);
			}
			else
				n = new ReluSyncedNeuron(false, false);
			n->neuron_age = 0;
			n->drinking_age = 20000;
			n->set_layer_number(current_layer_number);
			this->all_neurons.push_back(static_cast<SyncedNeuron*>(n));
			curr_layer.push_back(static_cast<SyncedNeuron*>(n));

			for (int synapse_idx = 0; synapse_idx < param_group.size(1); synapse_idx++) {
				SyncedNeuron *source_neuron;
				if (current_layer_number > 1)
					source_neuron = this->all_neuron_layers[current_layer_number-2][synapse_idx];
				else
					source_neuron = this->input_neurons[synapse_idx];
				auto new_synapse = new SyncedSynapse(source_neuron,
				                                     n,
				                                     param_group.index({neuron_idx, synapse_idx}).item<float>(),
				                                     step_size);
				this->all_synapses.push_back(new_synapse);
			}

		}

		if (current_layer_number < trained_model.parameters().size())
			this->all_neuron_layers.push_back(curr_layer);
		if (current_layer_number > trained_model.parameters().size()) {
			std::cout << "shouldnt happen" <<std::endl;
			exit(1);
		}
		current_layer_number += 1;
	}

}

PretrainedDenseNetwork::~PretrainedDenseNetwork() {

}

void PretrainedDenseNetwork::forward(std::vector<float> inp) {

//  std::cout << "Set inputs\n";

	this->set_input_values(inp);

//  std::cout << "Firing\n";

	std::for_each(
		std::execution::par_unseq,
		this->input_neurons.begin(),
		this->input_neurons.end(),
		[&](SyncedNeuron *n) {
		n->fire(this->time_step);
	});



	int counter = 0;
	for (auto LTU_neuron_list: this->all_neuron_layers) {
		counter++;
//    std::cout << "Updating values " << counter << "\n";
		std::for_each(
			std::execution::par_unseq,
			LTU_neuron_list.begin(),
			LTU_neuron_list.end(),
			[&](SyncedNeuron *n) {
			n->update_value(this->time_step);
		});

//    std::cout << "Firing " << counter << "\n";
		std::for_each(
			std::execution::par_unseq,
			LTU_neuron_list.begin(),
			LTU_neuron_list.end(),
			[&](SyncedNeuron *n) {
			n->fire(this->time_step);
		});

	}


//  std::cout << "Updating values output \n";
	std::for_each(
		std::execution::par_unseq,
		this->output_neurons.begin(),
		this->output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_value(this->time_step);
	});

//  std::cout << "Firing output \n";
	std::for_each(
		std::execution::par_unseq,
		this->output_neurons.begin(),
		this->output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->fire(this->time_step);
	});

//  std::cout << "Updating neuron utility \n";
	std::for_each(
		std::execution::par_unseq,
		this->all_neurons.begin(),
		this->all_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_utility();
	});

	this->time_step++;
}

void PretrainedDenseNetwork::backward(std::vector<float> target, bool update_weight) {
	this->introduce_targets(target);

	std::for_each(
		std::execution::par_unseq,
		output_neurons.begin(),
		output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->forward_gradients();
	});

	for (int layer = this->all_neuron_layers.size() - 1; layer >= 0; layer--) {
		std::for_each(
			std::execution::par_unseq,
			this->all_neuron_layers[layer].begin(),
			this->all_neuron_layers[layer].end(),
			[&](SyncedNeuron *n) {
			n->propagate_error();
		});

		std::for_each(
			std::execution::par_unseq,
			this->all_neuron_layers[layer].begin(),
			this->all_neuron_layers[layer].end(),
			[&](SyncedNeuron *n) {
			n->forward_gradients();
		});
	}
//  Calculate our credit

	std::for_each(
		std::execution::par_unseq,
		output_synapses.begin(),
		output_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_utility();
	});



	std::for_each(
		std::execution::par_unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->assign_credit();
	});

//  std::for_each(
//      std::execution::par_unseq,
//      output_synapses.begin(),
//      output_synapses.end(),
//      [&](SyncedSynapse *s) {
//        s->set_reinforce();
//      });

//  std::for_each(
//      std::execution::par_unseq,
//      all_neurons.begin(),
//      all_neurons.end(),
//      [&](SyncedNeuron *s) {
//        s->update_reinforcement();
//      });


//
////  Update our weights (based on either normal update or IDBD update
	if(update_weight) {
		std::for_each(
			std::execution::par_unseq,
			all_synapses.begin(),
			all_synapses.end(),
			[&](SyncedSynapse *s) {
			s->update_weight();
		});
	}
	if (this->time_step % 40 == 0) {
		std::for_each(
			this->all_neurons.begin(),
			this->all_neurons.end(),
			[&](SyncedNeuron *n) {
			n->mark_useless_weights();
		});

		std::for_each(
			this->all_neurons.begin(),
			this->all_neurons.end(),
			[&](SyncedNeuron *n) {
			n->prune_useless_weights();
		});

		int counter=0;
		for(int vector_ind = 0; vector_ind < this->all_neuron_layers.size(); vector_ind++) {
//    for (auto LTU_neuron_list: this->all_neuron_layers) {
			counter++;

			auto it_n = std::remove_if(this->all_neuron_layers[vector_ind].begin(),
			                           this->all_neuron_layers[vector_ind].end(),
			                           to_delete_synced_n);
			if (it_n != this->all_neuron_layers[vector_ind].end()) {
//        std::cout << "Deleting unused neurons\n";
//        std::cout << this->all_neuron_layers[vector_ind].size() << std::endl;
				this->all_neuron_layers[vector_ind].erase(it_n, this->all_neuron_layers[vector_ind].end());
//        std::cout << this->all_neuron_layers[vector_ind].size() << std::endl;
			}
		}
//      LTU_neuron_list.erase(it_n, LTU_neuron_list.end());
	}



	auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_synced_s);
	this->all_synapses.erase(it, this->all_synapses.end());

	it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_synced_s);
	this->output_synapses.erase(it, this->output_synapses.end());

	auto it_n_2 = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_synced_n);
	this->all_neurons.erase(it_n_2, this->all_neurons.end());
//    std::cout << "All neurons deleted\n";
//  }

}
