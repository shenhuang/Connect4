using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour {

	// Use this for initialization
	void Start () {
		//CreateConnect4AI ();
		//Connect4AIMoves ();
		//Connect4AILearns ();
		int[] sizes = { 2, 3, 8, 4, 7, 6, 3 };
		Network myNetwork = new Network (sizes);
		float[] a = { 1.2f, 2.3f };
		float[] a_prime = myNetwork.feedforward (a);
		myNetwork.backprop (a, a);
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

	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *																													 *
	 *																													 *
	 *																													 *
	 *											Artificial Neural Network												 *
	 *																													 *
	 *																													 *
	 *				This Artificial Neural Network is implemented based on Michael A. Nielsen's book:					 *
	 *										"Neural Networks and Deep Learning"											 *
	 *																													 *
	 *				The purpose of this implementation is to help creating a Neural Network based artificial			 *
	 *				intelligence for the game "Connect4" that will be demonstrated in a Machine Learning class			 *
	 *				in California State University, Northridge (COMP 496-ML).											 *
	 *																													 *
	 *				The copyright of the code belongs to Shen Huang, and is writted in C# specificly for Unity,			 *
	 *				which is the game engine used to implement the game that does not yet have many neural				 *
	 *				networks implemented.																				 *
	 *																					 								 *
	 *																					 								 *
	 *																					 		--- May 1, 2018			 *
	 *																					 								 *
	 *																					 								 *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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
		public static float[][]		biases;
		public static float[][][]	weights;
		//Identical matricies to store necessary information for back propagation.
		public static float[][]		nabla_b;
		public static float[][][]	nabla_w;

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
			biases = generateBiases(1);
			weights = generateWeights(1);
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

		public float[][] generateBiases (int method)
		{
			//Returns a biases array.
			//Method = 0: Returns a biases array of zeroes.
			//Method = 1; Returns a biases array of random numbers in normal distribution.
			//Since the index of arraries starts at 0, biases starts at layer 2,
			//Bias(Layer, Neuron) is represented as biases[Layer - 2][Neuron - 1].
			float[][] b = new float[num_layers - 1][];
			switch(method)
			{
				case 0:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeurons = sizes [layerIndex + 1];
						float[] b_layer = new float[nNeurons];
						for (int neuronIndex = 0; neuronIndex < nNeurons; neuronIndex++)
						{
							b_layer [neuronIndex] = 0;
						}
						b [layerIndex] = b_layer;
					}
					break;
				case 1:
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
			}
			return b;
		}

		public float[][][] generateWeights (int method)
		{
			//Returns a weight array.
			//Method = 0: Returns a weight array of zeroes.
			//Method = 1; Returns a weight array of random numbers in normal distribution.
			//Since the index of arraries starts at 0, weights starts at layer 1,
			//Weights(Layer, Neuron (From), Neuron (To)) is represented as weights[Layer - 1][Neuron (From) - 1][Neuron (To) - 1].
			float[][][] w = new float[num_layers][][];
			switch(method)
			{
				case 0:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[sizes [layerIndex]][];
						for (int fromNeuronIndex = 0; fromNeuronIndex < nNeuronsThisLayer; fromNeuronIndex++)
						{
							float[] weights_toNeurons = new float[nNeuronsNextLayer];
							for (int toNeuronIndex = 0; toNeuronIndex < nNeuronsNextLayer; toNeuronIndex++)
							{
								weights_toNeurons [toNeuronIndex] = 0;
							}
							w_layer [fromNeuronIndex] = weights_toNeurons;
						}
						w [layerIndex] = w_layer;
					}
					break;
				case 1:
					for (int layerIndex = 0; layerIndex < num_layers - 1; layerIndex++)
					{
						int nNeuronsThisLayer = sizes [layerIndex];
						int nNeuronsNextLayer = sizes [layerIndex + 1];
						float[][] w_layer = new float[sizes [layerIndex]][];
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
			//a' = sigmoid(sigma(wa + b)) with w, a and b as vectors in the previous layer.
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

		public void backprop (float[] x, float[] y)
		{
			//Modifies nabla_b and nabla_w to represent the gradient for the cost function C_x.
			nabla_b = generateBiases (0);
			nabla_w = generateWeights (0);
			//Feedforward using a' = sigmoid(sigma(wa + b)).
			float[]		zs = new float[num_layers - 1];			//List to store all the z vectors, layer by layer.			
			float[][]	activations = generateNeurons (0);		//List to store all the other activations, layer by layer.
			activations[0] = x;									//Initial activation equals to the input vector x.
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
						z += weights [layerIndex] [inputNeuronIndex] [outputNeuronIndex] * activations [layerIndex] [inputNeuronIndex] + biases [layerIndex] [outputNeuronIndex];
					}
					zs [layerIndex] = z;
					activations [layerIndex + 1] [outputNeuronIndex] = MathLibrary.sigmoid (z);
				}
			}
			//Backward pass.




		}

		public float[] cost_derivative()
		{
			int 
			float[] delta = new float[sizes[sizes.Length - 1]];


		}
	}

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
			return 1.0f/(1.0f + Mathf.Exp (-z));
		}

		public static float sigmoid_prime (float z)
		{
			return sigmoid (z) * (1 - sigmoid (z));
		}
	}

}
