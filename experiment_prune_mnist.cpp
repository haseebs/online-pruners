#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <memory>

#include <torch/script.h>
#include <torch/data.h>
#include <torch/nn.h>

#include "include/utils.h"
#include "include/nn/networks/pretrained_dense_network.h"
#include "include/experiment/Experiment.h"
#include "include/nn/utils.h"
#include "include/experiment/Metric.h"
#include "include/environments/mnist/mnist_reader.hpp"
#include "include/environments/mnist/mnist_utils.hpp"



int main(int argc, char *argv[]){


	float running_error = 6;
	float accuracy = 0.1;
	Experiment *my_experiment = new ExperimentJSON(argc, argv);

	Metric error_metric = Metric(my_experiment->database_name, "error_table",
	                             std::vector < std::string > {"step", "run", "error", "accuracy"},
	                             std::vector < std::string > {"int", "int", "real", "real"},
	                             std::vector < std::string > {"step", "run"});
	Metric error_metric_test = Metric(my_experiment->database_name, "test_set",
	                                  std::vector < std::string > {"step", "run", "accuracy", "mode"},
	                                  std::vector < std::string > {"int", "int", "real", "int"},
	                                  std::vector < std::string > {"step", "run", "mode"});


	torch::jit::script::Module trained_model;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		std::cout << "loading the torch model from: \t" << my_experiment->get_string_param("trained_model_path") << std::endl;
		trained_model = torch::jit::load(my_experiment->get_string_param("trained_model_path"));
	}
	catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	}

	PretrainedDenseNetwork network = PretrainedDenseNetwork(trained_model,
	                                                        my_experiment->get_float_param("step_size"),
	                                                        my_experiment->get_int_param("seed"),
	                                                        14*14,
	                                                        10,
	                                                        0.001);

	std::cout << "Step " << 0 << std::endl;

	std::cout << "Network confing\n";
	std::cout << "No\tSize\tSynapses\tOutput\n";
	for(int layer_no = 0; layer_no < network.all_neuron_layers.size(); layer_no++)
		std::cout <<  layer_no << "\t" << network.all_neuron_layers[layer_no].size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;
	std::cout <<  1000 << "\t" << network.output_neurons.size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;
	//TODO save the scaled mnist
	std::vector<std::vector<std::string> > error_logger;
	std::vector<std::vector<std::string> > error_logger_test;

	auto train_dataset =
		torch::data::datasets::MNIST("data/")
		.map(torch::data::transforms::Normalize<>(0.1307, 0.2801))
		.map(torch::data::transforms::Stack<>());
	int total_training_items = 60000;
	std::vector<std::vector<float> > images;
	std::vector<std::vector<float> > targets;

	for(int counter = 0; counter < total_training_items; counter++) {
		std::vector<float> x_temp;
		auto x = torch::nn::functional::interpolate(train_dataset.get_batch(counter).data,
		                                            torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({14,14})).mode(torch::kBilinear).align_corners(false));
		auto x_vec = x.reshape({14*14});
		for (int i = 0; i < 14*14; i++)
			x_temp.push_back(x_vec.index({i}).item<float>());
		images.push_back(x_temp);

		std::vector<float> y_vec;
		y_vec.push_back(train_dataset.get_batch(counter).target.item<float>());
		targets.push_back(y_vec);
	}

	std::mt19937 mt(my_experiment->get_int_param("seed"));

	int total_data_points = 60000;
	int total_steps = 0;
	bool training_phase = true;

  std::uniform_int_distribution<int> index_sampler(0, total_data_points - 1);

	for (int i = 0; i < my_experiment->get_int_param("steps"); i++) {
		total_steps++;
		int index = index_sampler(mt);
		auto x = images[index];
		float y_index = targets[index][0];
		std::vector<float> y(10);
		y[y_index] = 1;

		network.forward(x);
		auto prediction = network.read_output_values();
		float error = 0;
		for(int i = 0; i<prediction.size(); i++) {
			error += (prediction[i]-y[i])*(prediction[i]-y[i]);
		}
		running_error = running_error * 0.999 + 0.001 * sqrt(error);
		if(argmax(prediction) == y_index) {
			accuracy = accuracy*0.999 + 0.001;
		}
		else{
			accuracy*= 0.999;
		}


		network.backward(y, training_phase);
		if (i % 100 == 0) {
			std::vector<std::string> error;
			error.push_back(std::to_string(i));
			error.push_back(std::to_string(my_experiment->get_int_param("run")));
			error.push_back(std::to_string(running_error));
			error.push_back(std::to_string(accuracy));
			error_logger.push_back(error);

		}
		if(i % 10000 == 0) {
			std::cout << error_logger.size() << std::endl;
			error_metric.add_values(error_logger);
			error_logger.clear();
		}

		if (i % 1000 == 0) {
			std::cout << "Step " << i << std::endl;

			std::cout << "Network confing\n";
			std::cout << "No\tSize\tSynapses\tOutput\n";
			for(int layer_no = 0; layer_no < network.all_neuron_layers.size(); layer_no++)
				std::cout <<  layer_no << "\t" << network.all_neuron_layers[layer_no].size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;
      std::cout <<  1000 << "\t" << network.output_neurons.size() << "\t" << network.all_synapses.size() << "\t\t" << network.output_synapses.size() <<  std::endl;

			std::cout << "Running accuracy = " << accuracy << std::endl;
			std::cout << "GT " << y_index <<  " Pred = " << argmax(prediction) << std::endl;
			std::cout << " Target\n";
			print_vector(y);

			std::cout << " Prediction\n";
			print_vector(prediction);
			std::cout << "Running error = " << running_error << std::endl;
		}

	total_steps++;
	}
	error_metric.add_values(error_logger);
	error_logger.clear();
}
