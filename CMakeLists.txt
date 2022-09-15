#CC Build instructions
#module load python/3.7 and  module load scipy-stack and module load mariadb
#module load mariadb-connector-c
cmake_minimum_required(VERSION 3.14)
project(FlexibleNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#GCloud
#set(CMAKE_CXX_FLAGS "-O3 -lpthread -ldl -lm -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb -I/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/include/mariadb/mysql -L/home/khurramjaved/mariadb/mariadb-connector-c-3.1.13-ubuntu-focal-amd64/lib/mariadb/")

#Cedar
message( "Found environment variable SLURM_TMPDIR=$ENV{SLURM_TMPDIR}" )

set(CMAKE_CXX_FLAGS "-O3 -ltbb -lpthread")
find_package(Torch REQUIRED)

add_subdirectory(pybind11)
include_directories(pybind11/include/pybind11)

add_executable(loading-test loading-test.cpp)

add_executable(ModelExportTest src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
        include/nn/networks/pretrained_dense_network.h src/nn/networks/pretrained_dense_network.cpp
        include/json.hpp tests/cpp_mnist_prediction.cpp)

add_executable(MNISTPruning src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
        include/nn/networks/pretrained_dense_network.h src/nn/networks/pretrained_dense_network.cpp
        include/json.hpp experiment_prune_mnist.cpp)

add_executable(MNISTPruningNoPretrain src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
        include/nn/networks/dense_network.h src/nn/networks/dense_network.cpp
        include/json.hpp experiment_prune_mnist_nopretrained.cpp)

add_executable(SmallExample src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        include/nn/message.h src/nn/message.cpp include/nn/dynamic_elem.h src/nn/dynamic_elem.cpp
        include/nn/networks/synced_network.h src/nn/networks/synced_network.cpp
        include/nn/synced_neuron.h include/nn/synced_synapse.h src/nn/synced_neuron.cpp src/nn/synced_synapse.cpp
        include/nn/networks/small_network.h src/nn/networks/small_network.cpp
        include/json.hpp exp_linear.cpp)

pybind11_add_module(FlexibleNN src/utils.cpp include/utils.h include/experiment/Experiment.h src/experiment/Experiment.cpp include/experiment/Database.h
        src/experiment/Database.cpp include/experiment/Metric.h src/experiment/Metric.cpp include/nn/utils.h src/nn/utils.cpp
        pybind_interface/interface.cpp)


#Cedar
#target_link_libraries(FlexibleNN PRIVATE mariadb)

# FOR DEBUGGING
#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")

# For linking locally
target_link_libraries(FlexibleNN PUBLIC tbb)
target_link_libraries(FlexibleNN PUBLIC mariadb)

target_link_libraries(MNISTPruning PUBLIC tbb)
target_link_libraries(MNISTPruning PUBLIC mariadb)
target_link_libraries(MNISTPruning PUBLIC torch)

target_link_libraries(MNISTPruningNoPretrain PUBLIC tbb)
target_link_libraries(MNISTPruningNoPretrain PUBLIC mariadb)
target_link_libraries(MNISTPruningNoPretrain PUBLIC torch)

target_link_libraries(SmallExample PUBLIC tbb)
target_link_libraries(SmallExample PUBLIC mariadb)
target_link_libraries(SmallExample PUBLIC torch)

target_link_libraries(ModelExportTest PUBLIC tbb)
target_link_libraries(ModelExportTest PUBLIC mariadb)
target_link_libraries(ModelExportTest PUBLIC torch)

target_link_libraries(loading-test "${TORCH_LIBRARIES}")


project(FlexibleNN LANGUAGES CXX)
