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
                                               float utility_to_keep,
                                               float perc_prune,
                                               int min_synapses_to_keep,
                                               int prune_interval,
                                               int start_pruning_at,
                                               float trace_decay_rate) {

	this->mt.seed(seed);
	this->perc_prune = perc_prune;
	this->min_synapses_to_keep = min_synapses_to_keep;
	this->prune_interval = prune_interval;
	this->start_pruning_at = start_pruning_at;
	this->trace_decay_rate = trace_decay_rate;

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
				new_synapse->set_utility_to_keep(utility_to_keep);
				new_synapse->trace_decay_rate = trace_decay_rate;
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
	this->total_initial_synapses = this->all_synapses.size();

}

PretrainedDenseNetwork::~PretrainedDenseNetwork() {
}



void PretrainedDenseNetwork::forward(std::vector<float> inp) {

//  std::cout << "Set inputs\n";

	this->set_input_values(inp);

//  std::cout << "Firing\n";

	std::for_each(
		std::execution::unseq,
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
			std::execution::unseq,
			LTU_neuron_list.begin(),
			LTU_neuron_list.end(),
			[&](SyncedNeuron *n) {
			n->update_value(this->time_step);
		});

//    std::cout << "Firing " << counter << "\n";
		std::for_each(
			std::execution::unseq,
			LTU_neuron_list.begin(),
			LTU_neuron_list.end(),
			[&](SyncedNeuron *n) {
			n->fire(this->time_step);
		});

	}


//  std::cout << "Updating values output \n";
	std::for_each(
		std::execution::unseq,
		this->output_neurons.begin(),
		this->output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_value(this->time_step);
	});

//  std::cout << "Firing output \n";
	std::for_each(
		std::execution::unseq,
		this->output_neurons.begin(),
		this->output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->fire(this->time_step);
	});

//  std::cout << "Updating neuron utility \n";

	this->time_step++;
}

void PretrainedDenseNetwork::backward(std::vector<float> target) {
	this->introduce_targets(target);

	std::for_each(
		std::execution::unseq,
		output_neurons.begin(),
		output_neurons.end(),
		[&](SyncedNeuron *n) {
		n->forward_gradients();
	});

	for (int layer = this->all_neuron_layers.size() - 1; layer >= 0; layer--) {
		std::for_each(
			std::execution::unseq,
			this->all_neuron_layers[layer].begin(),
			this->all_neuron_layers[layer].end(),
			[&](SyncedNeuron *n) {
			n->propagate_error();
		});

		std::for_each(
			std::execution::unseq,
			this->all_neuron_layers[layer].begin(),
			this->all_neuron_layers[layer].end(),
			[&](SyncedNeuron *n) {
			n->forward_gradients();
		});
	}

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->assign_credit();
	});
}


void PretrainedDenseNetwork::update_weights() {
	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_weight();
	});
}


void PretrainedDenseNetwork::update_utility_estimates(std::string pruner){
	if (pruner == "utility_propagation")
		this->update_utility_propagation_estimates();
	else if (pruner == "activation_trace")
		this->update_activation_trace_estimates();
	else if (pruner == "dropout_utility_estimator")
		this->update_dropout_utility_estimates();
}


void PretrainedDenseNetwork::update_activation_trace_estimates(){
	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_activation_trace();
	});
}


void PretrainedDenseNetwork::update_utility_propagation_estimates(){
	std::for_each(
		std::execution::unseq,
		this->all_neurons.begin(),
		this->all_neurons.end(),
		[&](SyncedNeuron *n) {
		n->update_utility();
	});

	std::for_each(
		std::execution::unseq,
		all_synapses.begin(),
		all_synapses.end(),
		[&](SyncedSynapse *s) {
		s->update_utility();
	});
}


void PretrainedDenseNetwork::update_dropout_utility_estimates(std::vector<float> inp,
                                                              std::vector<float> normal_predictions,
                                                              float dropout_perc){
	int total_dropped_synapses = this->all_synapses.size() * dropout_perc;
	if (total_dropped_synapses < 1)
		total_dropped_synapses = 1;

	std::vector<SyncedSynapse *> synapses_to_drop;
	std::sample(this->all_synapses.begin(),
	            this->all_synapses.end(),
	            std::back_inserter(synapses_to_drop),
	            total_dropped_synapses,
	            this->mt);

	for (int i = 0; i < total_dropped_synapses; i++)
		synapses_to_drop[i]->is_dropped_out= true;
	//TODO this is bugged with non-zero step-sizes
	this->forward(inp);
	this->time_step--; //this forward pass is not an actual step
	auto dropout_predictions = this->read_output_values();

	float sum_of_differences = 0;
	for (int i = 0; i < dropout_predictions.size(); i++)
		sum_of_differences += fabs(normal_predictions[i] - dropout_predictions[i]);

	for (int i = 0; i < total_dropped_synapses; i++) {
		synapses_to_drop[i]->dropout_utility_estimate = this->trace_decay_rate * synapses_to_drop[i]->dropout_utility_estimate + (1-this->trace_decay_rate) * sum_of_differences;
		synapses_to_drop[i]->is_dropped_out = false;
	}
}


void PretrainedDenseNetwork::prune_using_dropout_utility_estimator() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return fabs(a->dropout_utility_estimate) < fabs(b->dropout_utility_estimate);
	} );

	for (int i = 0; i < total_removals; i++)
		all_synapses_copy[i]->is_useless = true;
}


void PretrainedDenseNetwork::prune_using_trace_of_activation_magnitude() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return fabs(a->activation_trace) < fabs(b->activation_trace);
	} );

	for (int i = 0; i < total_removals; i++)
		all_synapses_copy[i]->is_useless = true;
}


void PretrainedDenseNetwork::prune_using_weight_magnitude_pruner() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return fabs(a->weight) < fabs(b->weight);
	} );

	for (int i = 0; i < total_removals; i++) {
		all_synapses_copy[i]->is_useless = true;
	}
}


void PretrainedDenseNetwork::prune_using_random_pruner() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> synapses_to_remove;
	std::sample(this->all_synapses.begin(),
	            this->all_synapses.end(),
	            std::back_inserter(synapses_to_remove),
	            total_removals,
	            this->mt);

	for (int i = 0; i < total_removals; i++) {
		synapses_to_remove[i]->is_useless = true;
	}
}


void PretrainedDenseNetwork::prune_using_utility_propoagation() {
	if (all_synapses.size() < this->min_synapses_to_keep)
		return;

	int total_removals = 1;
	std::vector<SyncedSynapse *> all_synapses_copy(this->all_synapses);

	std::nth_element(all_synapses_copy.begin(),
	                 all_synapses_copy.begin() + total_removals,
	                 all_synapses_copy.end(),
	                 []( auto a, auto b ) {
		return a->synapse_utility < b->synapse_utility;
	} );
	for (int i = 0; i < total_removals; i++) {
		all_synapses_copy[i]->is_useless = true;
	}
}


int PretrainedDenseNetwork::get_current_synapse_schedule() {
	return std::max(int(this->min_synapses_to_keep),
	                int( this->total_initial_synapses - ( (this->time_step - this->start_pruning_at) / this->prune_interval )));
}


void PretrainedDenseNetwork::prune_weights(std::string pruner){
	if (this->time_step > this->prune_interval && this->time_step > this->start_pruning_at && this->time_step % this->prune_interval == 0) {
		if (this->all_synapses.size() > this->get_current_synapse_schedule()) {
			//std::cout << "pruuuuuuuuuuuuuuuuuuuuuuuune" << std::endl;
			if (pruner == "utility_propagation")
				this->prune_using_utility_propoagation();
			else if (pruner == "random")
				this->prune_using_random_pruner();
			else if (pruner == "weight_magnitude")
				this->prune_using_weight_magnitude_pruner();
			else if (pruner == "activation_trace")
				this->prune_using_trace_of_activation_magnitude();
			else if (pruner == "dropout_utility_estimator")
				this->prune_using_dropout_utility_estimator();
			else if (pruner != "none") {
				std::cout << "Invalid pruner specified" << std::endl;
				exit(1);
			}

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
				counter++;

				auto it_n = std::remove_if(this->all_neuron_layers[vector_ind].begin(),
				                           this->all_neuron_layers[vector_ind].end(),
				                           to_delete_synced_n);
				if (it_n != this->all_neuron_layers[vector_ind].end()) {
					this->all_neuron_layers[vector_ind].erase(it_n, this->all_neuron_layers[vector_ind].end());
				}
			}

			auto it = std::remove_if(this->all_synapses.begin(), this->all_synapses.end(), to_delete_synced_s);
			this->all_synapses.erase(it, this->all_synapses.end());

			it = std::remove_if(this->output_synapses.begin(), this->output_synapses.end(), to_delete_synced_s);
			this->output_synapses.erase(it, this->output_synapses.end());

			auto it_n_2 = std::remove_if(this->all_neurons.begin(), this->all_neurons.end(), to_delete_synced_n);
			this->all_neurons.erase(it_n_2, this->all_neurons.end());
		}
	}
}
