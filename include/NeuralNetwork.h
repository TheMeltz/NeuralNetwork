#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "Matrix.h"

class NeuralNetwork {
public:
	NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes);

	Matrix predict(const Matrix& input) const;
	void train(const Matrix& trainingInput, const Matrix& target, int epochs, double learningRate);
private:
	Matrix weightsInputHidden;
    Matrix weightsHiddenOutput;
    Matrix biasHidden;
    Matrix biasOutput;
};

#endif