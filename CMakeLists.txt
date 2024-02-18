cmake_minimum_required(VERSION 3.10)

include_directories(include)

project(NeuralNetwork)
add_executable(NeuralNetwork 
	src/main.cpp
	src/Matrix.cpp
	src/NeuralNetwork.cpp
)

target_compile_features(NeuralNetwork PRIVATE cxx_std_11)