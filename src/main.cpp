/*
	Uma rede neural que prevê a chance de aprovação em um concurso, baseado em:
	- Horas estudadas (Nos últimos 6 meses);
	- Horas dormidas;
	- Nota simulado;
	
	NeuralNetwork StudentSuccessEvaluator(3, 2, 1);
	Matrix chances = StudentSuccessEvaluator.predict({
		{6 * 30 * 5.0, 8.0, 10.0},
		{6 * 30 * 2.0, 6.0, 8.0},
		{6 * 30 * 1.0, 3.0, 2.5}
	});
	chances.print();

	Resultado esperado:
	{
		{1},
		{0.8},
		{0.01}
	}

*/

#include <iostream>
#include "NeuralNetwork.h"
#include "Matrix.h"

#define INPUT_NODES 3
#define HIDDEN_NODES 2
#define OUTPUT_NODES 1

#define EPOCHS 1000
#define LEARNING_RATE 0.01

int main() {
	NeuralNetwork StudentSuccessEvaluator(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);

	Matrix trainingInput = {
		{6 * 30 * 5.0, 8.0, 10.0},
		{6 * 30 * 2.0, 6.0, 8.0},
		{6 * 30 * 1.0, 3.0, 2.5}
	};

	Matrix trainingTarget = {
		{1},
		{0.8},
		{0.01}
	};

	StudentSuccessEvaluator.train(trainingInput, trainingTarget, EPOCHS, LEARNING_RATE);

	Matrix chances = StudentSuccessEvaluator.predict({
		{6 * 30 * 5.0, 8.0, 10.0},
		{6 * 30 * 2.0, 6.0, 8.0},
		{6 * 30 * 1.0, 3.0, 2.5}
	});

	chances.print();

	return 0;
}