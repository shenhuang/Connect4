using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour
{
	// Use this for initialization
	void Start () {
		//CreateConnect4AI ();
		//Connect4AIMoves ();
		//Connect4AILearns ();
		int[] sizes = { 3, 5, 2, 3, 4 };
		Network myNetwork = new Network (sizes);
		float[] x = { 3.0f, 1.7f, 4.0f };
		float[] output = myNetwork.feedforward (x);
		//Debug.Log ("O: " + output[0] + ", " + output[1] + ", " + output[2] + ", " + output[3]);
		float[][][] miniBatch = new float[5][][];
		for (int index = 0; index < 5; index++)
		{
			float[] x_batch = { 1.0f, 0.0f, 1.0f };
			float[] y_batch = { 5.0f, 5.0f, 0.0f, 5.0f };
			float[][] batches = new float[2][];
			batches [0] = x_batch;
			batches [1] = y_batch;
			miniBatch[index] = batches;
		}
		for (int index = 0; index < 100; index++)
		{
			myNetwork.update_mini_batch (miniBatch, 0.15f);
			output = myNetwork.feedforward (x);
			//Debug.Log ("O: " + output[0] + ", " + output[1] + ", " + output[2] + ", " + output[3]);
		}
	}
	
	// Update is called once per frame
	void Update () {
		
	}

	void CreateConnect4AI ()
	{
		
	}

	Vector2 Connect4AIMoves ()
	{
		Vector2 move = new Vector2 (-1, -1);



		return move;
	}

	void Connect4AILearns ()
	{
		
	}

	/* * * * * * * * * * * * * * * * * * * * *
	 *										 *
	 *										 *
	 *										 *
	 *					Tutor				 *
	 *										 *
	 *										 *
	 *										 *
	 * * * * * * * * * * * * * * * * * * * * */

	//A tutor that attempts to teach generate training sets for the Artificial Neural Network.



	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *																										 *
	 *																										 *
	 *																										 *
	 *									Artificial Neural Network											 *
	 *																										 *
	 *																										 *
	 *		This Artificial Neural Network is implemented based on Michael A. Nielsen's book:				 *
	 *								"Neural Networks and Deep Learning"										 *
	 *																										 *
	 *		The purpose of this implementation is to help creating a Neural Network based artificial		 *
	 *		intelligence for the game "Connect4" that will be demonstrated in a Machine Learning class		 *
	 *		in California State University, Northridge (COMP 496-ML).										 *
	 *																										 *
	 *		The copyright of the code belongs to Shen Huang, and is writted in C# specificly for Unity,		 *
	 *		which is the game engine used to implement the game that does not yet have many neural			 *
	 *		networks implemented.																			 *
	 *																					 					 *
	 *																					 					 *
	 *																			 		--- May 1, 2018		 *
	 *																					 					 *
	 *																					 					 *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

	//Artificial Neural Network Class.

	public class Network
	{
		public static int 			num_layers;
		public static int[] 		sizes;
		//The biases and weights are represented by multi-dimentional vectors.
		//Due to the property where the index of the vector starts at 0, and the inclusion of bias starts from layer 2,
		//b[l][j], which is the bias of the jth neuron in layer l,
		//is represented as biases[l - 1][j - 2],
		//w[l][j][k] which is the weight from the jth neuron in layer l to the kth neuron in layer l + 1,
		//is represented as weights[l - 1][j - 1][k - 1].
		//This representation is different from the way Michael A. Nielsen implemented it inside his tutorial,
		//this is because the libraries for matrix calculations in Unity is not as matured as in Python,
		//corresponding changes has been made inside the code to adapt to this twitch.
		public static float[][]		biases;
		public static float[][][]	weights;
		//Identical matricies to store necessary information for back propagation.
		public static float[][]		nabla_b;
		public static float[][][]	nabla_w;
		public static float[][]		nabla_b_total;
		public static float[][][]	nabla_w_total;

		/* * * * * * * * * * * * * * * * * * * * *
		 *										 *
		 *										 *
		 *										 *
		 *				Construction			 *
		 *										 *
		 *										 *
		 *										 *
		 * * * * * * * * * * * * * * * * * * * * */

		public Network (int[] networkSizes)
		{
			//Constructor of the neural network.
			//networkSizes is an array which specifies the number of neurons in each layer.
			//For example, [2, 3, 1] would specify an Artificial Neural Network that has
			//2 neurons in first layer, 3 in second and 1 the final layer.
			num_layers = networkSizes.Length;
			sizes = networkSizes;
			biases = generateBiases(0, 0.0f);
			weights = generateWeights(0, 0.0f);
		}

		public float[][] generateNeurons (float value)
		{
			//Returns an array representing the neurons with the given value.
			//Since the index of arraries starts at 0,
			//Neuron(Layer, Neuron) is represented as neurons[Layer - 1][Neuron - 1].
			float[][] n = new float[num_layers][];
			for (int layerIndex = 0; layerIndex < num_layers; layerIndex++)
			{
				int nNeurons = sizes [layerIndex];
				float[] n_layer = new float[nNeurons];
				for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
				{
					n_layer [neuronIndex] = 0;
				}
				n [layerIndex] = n_layer;
			}
			return n;
		}

		public float[][] generateBiases (int method, float value)
		{
			//Returns a biases array.
			//Method = 0: Returns a biases array of random numbers in normal distribution.
			//Method = -1: Returns uniformed arraries of value indicated by value.
			//Since the index of arraries starts at 0, biases starts at layer 2,
			//Bias(Layer, Neuron) is represented as biases[Layer - 2][Neuron - 1].
			float[][] b = new float[num_layers - 1][];
			switch (method)
			{
				case 0:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = MathLibrary.randomGaussianDistributedNumber (0, 1);
						}
						b [layerIndex] = b_layer;
					}
					break;
				case -1:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = value;
						}
						b [layerIndex] = b_layer;
					}
					break;
			}
			return b;
		}

		public float[][][] generateWeights (int method, float value)
		{
			//Returns a weight array.
			//Method = 0: Returns a weight array of random numbers in normal distribution.
			//Method = -1: Returns uniformed arraries of value indicated by method.
			//Since the index of arraries starts at 0, weights starts at layer 1,
			//Weights(Layer (From), Neuron (From), Neuron (To)) is represented as weights[Layer (From) - 1][Neuron (From) - 1][Neuron (To) - 1].
			float[][][] w = new float[num_layers - 1][][];
			switch (method)
			{
				case 0:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[nNeuronsThisLayer][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = MathLibrary.randomGaussianDistributedNumber (0, 1);
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
				case -1:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[nNeuronsThisLayer][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = value;
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
			}
			return w;
		}

		/* * * * * * * * * * * * * * * * * * * * *
		 *										 *
		 *										 *
		 *										 *
		 *				Utilization				 *
		 *										 *
		 *										 *
		 *										 *
		 * * * * * * * * * * * * * * * * * * * * */

		public float[] feedforward (float[] a)
		{
			//Generates the output vector a' given the input vector a,
			//where the activation vector a' in the new layer is calculated with:
			//a' = sigmoid(sigma(z)) where z = w * a + b,
			//with w, a and b as vectors in the previous layer.
			for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
			{
				int nInputNeurons = sizes [layerIndex];
				int nOutputNeurons = sizes [layerIndex + 1];
				float[] a_prime = new float[nOutputNeurons];
				for (int outputNeuronIndex = 0; outputNeuronIndex < nOutputNeurons; outputNeuronIndex++)
				{
					float z = 0;
					for (int inputNeuronIndex = 0; inputNeuronIndex < nInputNeurons; inputNeuronIndex++)
					{
						z += weights [layerIndex] [inputNeuronIndex] [outputNeuronIndex] * a [inputNeuronIndex] + biases [layerIndex] [outputNeuronIndex];
					}
					a_prime [outputNeuronIndex] = MathLibrary.sigmoid (z);
				}
				a = a_prime;
			}
			return a;
		}

		/* * * * * * * * * * * * * * * * * * * * *
		 *										 *
		 *										 *
		 *										 *
		 *				Training				 *
		 *										 *
		 *										 *
		 *										 *
		 * * * * * * * * * * * * * * * * * * * * */

		public void update_mini_batch(float[][][] mini_batch, float eta)
		{
			//Update the network's weights and biases by applying gradient
			//descent using backpropagation to a signle mini batch.
			//The mini_batch is a list of x and y vectors represented as:
			//x[i] = mini_batch[i][0] and y[i] = mini_batch[i][1].
			//eta is the learning rate.
			float[][] 	nabla_b_total = generateBiases (-1, 0.0f);
			float[][][]	nabla_w_total = generateWeights (-1, 0.0f);
			for (int index = 0; index < mini_batch.Length; index++)
			{
				backprop(mini_batch[index][0], mini_batch[index][1]);
				nabla_b_total = operate_biases(nabla_b_total, nabla_b, 0);
				nabla_w_total = operate_weights(nabla_w_total, nabla_w, 0);
			}
			float 		average_learning_rate = eta/mini_batch.Length;
			float[][]	delta_biases = generateBiases (-1, average_learning_rate);
			float[][][]	delta_weights = generateWeights (-1, average_learning_rate);
			delta_biases = operate_biases(delta_biases, nabla_b_total, 2);
			delta_weights = operate_weights(delta_weights, nabla_w_total, 2);
			biases = operate_biases (biases, delta_biases, 1);
			weights = operate_weights (weights, delta_weights, 1);
		}

		public void backprop (float[] x, float[] y)
		{
			//Updates nabla_b and nabla_w to represent the gradient for the cost function C_x.
			nabla_b = generateBiases (-1, 0.0f);
			nabla_w = generateWeights (-1, 0.0f);
			//Feedforward using a' = sigmoid(sigma(wa + b)),
			//while keeping the activations for each layer.
			float[][]	zs = new float[num_layers - 1][];		//List to store all the z vectors, layer by layer.			
			float[][]	activations = generateNeurons (0);		//List to store all the other activations, layer by layer.
			activations[0] = x;									//Initial activation equals to the input vector x.
			for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
			{
				int nInputNeurons = sizes [layerIndex];
				int nOutputNeurons = sizes [layerIndex + 1];
				float[] z = new float[nOutputNeurons];
				for (int outputNeuronIndex = 0; outputNeuronIndex < nOutputNeurons; outputNeuronIndex++)
				{
					float sum = 0;
					for (int inputNeuronIndex = 0; inputNeuronIndex < nInputNeurons; inputNeuronIndex++)
					{
						sum += weights [layerIndex] [inputNeuronIndex] [outputNeuronIndex] * activations [layerIndex] [inputNeuronIndex] + biases [layerIndex] [outputNeuronIndex];
					}
					z [outputNeuronIndex] = sum;
					activations [layerIndex + 1] [outputNeuronIndex] = MathLibrary.sigmoid (z [outputNeuronIndex]);
				}
				zs [layerIndex] = z;
			}
			//Backward pass.
			//Generate nabla_b and nabla_w for the first layer.
			//delta of the first layer is calculated by:
			//delta = dC/da * sigmoid (z), where for quadratic cost function:
			//C = 1/2 * sigma(y - a)^2 -> dC/da = a - y.
			float[] delta = new float[y.Length];
			for (int index = 0; index < y.Length; index++)
			{
				delta [index] = (activations [num_layers - 1] [index] - y [index]) * MathLibrary.sigmoid_prime(zs [num_layers - 2] [index]);
			}
			nabla_b [num_layers - 2] = delta;
			nabla_w [num_layers - 2] = calculate_nabla_w (activations [num_layers - 2], delta);
			//Generate nabla_b and nabla_w for the rest of the layers.
			for (int l = num_layers - 3; l >= 0; l--)
			{
				delta = calculate_delta (weights [l + 1], delta, zs [l]);
				nabla_b [l] = delta;
				nabla_w [l] = calculate_nabla_w (activations [l], delta);
			}
		}

		public float[] calculate_delta (float[][] weights, float[] delta, float[] z)
		{
			//Returns a layer of the delta vector, where for this specific structure of w:
			//delta[l] = (transpose(w[l]) * delta [l + 1]) * sigmoid_prime(z[l]).
			int nInputNeurons = weights.Length;
			int nOutputNeurons = weights [0].Length;
			float[] new_delta = new float[nInputNeurons];
			for (int inputNeuronIndex = 0; inputNeuronIndex < nInputNeurons; inputNeuronIndex++)
			{
				new_delta [inputNeuronIndex] = 0;
				for (int outputNeuronIndex = 0; outputNeuronIndex < nOutputNeurons; outputNeuronIndex++)
				{
					new_delta [inputNeuronIndex] += weights [inputNeuronIndex] [outputNeuronIndex] * delta [outputNeuronIndex];
				}
				new_delta [inputNeuronIndex] *= MathLibrary.sigmoid_prime(z [inputNeuronIndex]);
			}
			return new_delta;
		}

		public float[][] calculate_nabla_w (float[] previous_activations, float[] delta)
		{
			//Returns a layer of the nabula_w vector, where:
			//nabla_w[l][j][k] = activations[l - 1][j] * delta[l][k].
			float[][] nabla_w_layer = new float [previous_activations.Length] [];
			for (int j = 0; j < previous_activations.Length; j++)
			{
				float[] nabla_w_vector = new float[delta.Length];
				for (int k = 0; k < delta.Length; k++)
				{
					nabla_w_vector [k] = previous_activations [j] * delta [k];
				}
				nabla_w_layer [j] = nabla_w_vector;
			}
			return nabla_w_layer;
		}

		public float[][] operate_biases (float[][] biases1, float[][] biases2, int operation)
		{
			//0:Adds, 1:Subtracts, 3:Multiplies, 4:Divides bias1 by bias2.
			//Return the result as b.
			float[][] b = new float[num_layers - 1][];
			switch(operation)
			{
				case(0):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = biases1 [layerIndex] [neuronIndex] + biases2 [layerIndex] [neuronIndex];
						}
						b [layerIndex] = b_layer;
					}
					break;
				case(1):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = biases1 [layerIndex] [neuronIndex] - biases2 [layerIndex] [neuronIndex];
						}
						b [layerIndex] = b_layer;
					}
					break;
				case(2):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = biases1 [layerIndex] [neuronIndex] * biases2 [layerIndex] [neuronIndex];
						}
						b [layerIndex] = b_layer;
					}
					break;
				case(3):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = biases1 [layerIndex] [neuronIndex] / biases2 [layerIndex] [neuronIndex];
						}
						b [layerIndex] = b_layer;
					}
					break;
			}
			return b;
		}

		public float[][][] operate_weights (float[][][] weights1, float[][][] weights2, int operation)
		{
			//0:Adds, 1:Subtracts, 3:Multiplies, 4:Divides weight1 by weight2.
			//Return the result as w.
			float[][][] w = new float[num_layers - 1][][];
			switch (operation)
			{
				case(0):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[nNeuronsThisLayer][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = weights1 [layerIndex] [fromNeuronIndex] [toNeuronIndex] + weights2 [layerIndex] [fromNeuronIndex] [toNeuronIndex];
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
				case(1):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[nNeuronsThisLayer][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = weights1 [layerIndex] [fromNeuronIndex] [toNeuronIndex] + weights2 [layerIndex] [fromNeuronIndex] [toNeuronIndex];
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
				case(2):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[nNeuronsThisLayer][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = weights1 [layerIndex] [fromNeuronIndex] [toNeuronIndex] + weights2 [layerIndex] [fromNeuronIndex] [toNeuronIndex];
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
				case(3):
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[nNeuronsThisLayer][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = weights1 [layerIndex] [fromNeuronIndex] [toNeuronIndex] + weights2 [layerIndex] [fromNeuronIndex] [toNeuronIndex];
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
			}
			return w;
		}

		/* * * * * * * * * * * * * * * * * * * * *
		 *										 *
		 *										 *
		 *										 *
		 *				Accessories				 *
		 *										 *
		 *										 *
		 *										 *
		 * * * * * * * * * * * * * * * * * * * * */

		public string structureToString (float[][] item)
		{
			//Converts structures similiar to a neural network to String.
			string output = "Contains:\n";
			int nLayers = item.Length;
			for (int layerIndex = 0; layerIndex < nLayers; layerIndex++)
			{
				output += "Layer" + (layerIndex + 1) + ":\n";
				int nNeurons = item [layerIndex].Length;
				for(int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
				{
					output += item [layerIndex] [neuronIndex] + "| ";
				}
				output += "\n";
			}
			return output;
		}

		public string structureToString (float[][][] item)
		{
			//Converts structures similiar to a weight matrix of the neural network to String.
			string output = "Contains:\n";
			int nLayers = item.Length;
			for (int layerIndex = 0; layerIndex < nLayers; layerIndex++)
			{
				output += "Layer" + (layerIndex + 1) + ":\n";
				int nNeurons = item [layerIndex].Length;
				for(int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
				{
					int nNextNeurons = item [layerIndex] [neuronIndex].Length;
					for (int nextNeuronIndex = 0; nextNeuronIndex < nNextNeurons; nextNeuronIndex++)
					{
						output += item [layerIndex] [neuronIndex] [nextNeuronIndex] + "| ";
					}
					output += "\n";
				}
				output += "\n";
			}
			return output;			
		}
	}

	/* * * * * * * * * * * * * * * * * * * * *
	 *										 *
	 *										 *
	 *										 *
	 *				Math Library			 *
	 *										 *
	 *										 *
	 *										 *
	 * * * * * * * * * * * * * * * * * * * * */

	//Necessary math features that are not avaliable in Unity.

	public class MathLibrary
	{
		public static float randomGaussianDistributedNumber (float mean, float stdDev)
		{
			//Generate a random number following Gaussian distribution utilizing Box-Muller transform.
			//U1 and U2 are independ random variables that are form the same rectangular density function in the interval (0, 1).
			float u1 = 1.0f - Random.Range (0.0f, 1.0f);
			float u2 = 1.0f - Random.Range (0.0f, 1.0f);
			//Z1 is a random vairable with a standard normal distribution,
			//Z1 is the real part of the polar coordinate indicated by R and Omega.
			//which is calculated with R * sin (Omega), where:
			//R^2 = -2 * ln(U1)
			//Omega = 2 * PI * U2
			//Which translates into Z1 = sqrt(-2 * ln(U1)) * sin(2 * PI * U2)).
			float Z1 = Mathf.Sqrt (-2.0f * Mathf.Log (u1)) * Mathf.Sin (2.0f * Mathf.PI * u2);
			float randNormal = mean + stdDev * Z1;
			return randNormal;
		}

		public static float sigmoid (float z)
		{
			//Sigmoid, a smoothed out step function,
			//is defined by 1/(1 + e^(-z)).
			return 1.0f / (1.0f + Mathf.Exp (-z));
		}

		public static float sigmoid_prime (float z)
		{
			return sigmoid (z) * (1 - sigmoid (z));
		}
	}
}