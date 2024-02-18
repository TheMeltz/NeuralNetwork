#include "NeuralNetwork.h"
#include "Matrix.h"

double ReLU(double x) {
	return (x > 0) ? x : 0;
}

double ReLU_derivative(double x) {
	return (x > 0) ? 1 : 0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize) : 
	  weightsInputHidden(inputSize, hiddenSize),
      weightsHiddenOutput(hiddenSize, outputSize),
      biasHidden(inputSize, hiddenSize),
      biasOutput(inputSize, outputSize) {
	weightsInputHidden.randomize();
    weightsHiddenOutput.randomize();

    biasHidden.randomize();
    biasOutput.randomize();
}

Matrix NeuralNetwork::predict(const Matrix& input) const {
	Matrix hiddenLayer = Matrix::dot(input.normalize(), weightsInputHidden) + biasHidden;
	hiddenLayer.map([&](int row, int col, double element) {
		return ReLU(element);
	});

	Matrix outputLayer = Matrix::dot(hiddenLayer, weightsHiddenOutput) + biasOutput;
	outputLayer.map([&](int row, int col, double element) {
		return sigmoid(element);
	});

	return outputLayer;
}

void NeuralNetwork::train(const Matrix& trainingInput, const Matrix& target, int epochs, double learningRate) {
	for (int epoch = 0; epoch < epochs; ++epoch) {
		Matrix hiddenLayer = Matrix::dot(trainingInput.normalize(), weightsInputHidden) + biasHidden;
		hiddenLayer.map([&](int row, int col, double element) {
			return ReLU(element);
		});

		Matrix outputLayer = Matrix::dot(hiddenLayer, weightsHiddenOutput) + biasOutput;
		outputLayer.map([&](int row, int col, double element) {
			return sigmoid(element);
		});

		try {
			Matrix outputError = outputLayer - target;
			Matrix error = target - outputLayer;
			double MSE = error.hadamard(error).getValue(0, 0) / (2 * trainingInput.getRows());

			Matrix derivativeOutputError = outputLayer;
			derivativeOutputError.map([&](int row, int col, double element) {
				return sigmoid_derivative(element);
			});
			Matrix outputDelta = outputError.hadamard(derivativeOutputError);

			Matrix hiddenError = Matrix::dot(outputDelta, weightsHiddenOutput.T());
			Matrix hiddenDerivative = hiddenLayer;
			hiddenDerivative.map([&](int row, int col, double element) {
				return ReLU_derivative(element);
			});
			Matrix hiddenDelta = hiddenError.hadamard(hiddenDerivative);

			weightsHiddenOutput = weightsHiddenOutput - Matrix::dot(hiddenLayer.T(), outputDelta) * (learningRate / trainingInput.getRows());
	        biasOutput = biasOutput - outputDelta;  // Subtrai elemento por elemento
			biasOutput = biasOutput - (outputDelta * (learningRate / trainingInput.getRows()));

	        weightsInputHidden = weightsInputHidden - Matrix::dot(trainingInput.T(), hiddenDelta) * (learningRate / trainingInput.getRows());
	        biasHidden = biasHidden - hiddenDelta;
	        biasHidden = biasHidden - (hiddenDelta * (learningRate / trainingInput.getRows()));

	        std::cout << "Epoch " << epoch << " MSE: " << MSE << std::endl;
		} catch (const std::invalid_argument& e) {
			std::cerr << "Error: " << e.what() << std::endl;
		}
		
	}
}