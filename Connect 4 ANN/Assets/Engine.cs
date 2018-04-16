using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Engine : MonoBehaviour {

	//Piece game objects.
	public GameObject pieceRed;
	public GameObject pieceBlue;

	//Piece positioning variables.
	public float xPieceOffset;
	public float yPieceOffset;
	public float xPieceScale;
	public float yPieceScale;

	//A matrix representing the board.
	//-1 Indicates Empty.
	//0 Indicates Red.
	//1 Indicates Blue.
	public int[,] boardMatrix;

	// Use this for initialization
	void Start () {

		xPieceOffset = -3.0f;
		yPieceOffset = -2.51f;
		xPieceScale = 1.0f;
		yPieceScale = 1.0f;
		resetMatrix ();
		updateBoard (boardMatrix);
	}
	
	// Update is called once per frame
	void Update ()
	{
		
	}

	/* * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *													 *
	 *													 *
	 *													 *
	 *							UI						 *
	 *													 *
	 *													 *
	 *													 *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * */

	//Put piece onto the given location.
	//Bottom Left is (0, 0), top right is (6, 5).
	//0 Indicates Red.
	//1 Indicates Blue.
	void putPiece(int x, int y, int pieceColor)
	{
		float xPosition = x * xPieceScale + xPieceOffset + transform.position.x;
		float yPosition = y * yPieceScale + yPieceOffset + transform.position.y;
		Vector3 putPosition = new Vector3 (xPosition, yPosition, 0);
		switch (pieceColor)
		{
		case 0:
				GameObject.Instantiate (pieceRed, putPosition, Quaternion.identity);
				break;
			case 1:
				GameObject.Instantiate (pieceBlue, putPosition, Quaternion.identity);
				break;
		}
	}

	/* * * * * * * * * * * * * * * * * * * * * * * * * * *
	 *													 *
	 *													 *
	 *													 *
	 *							AI						 *
	 *													 *
	 *													 *
	 *													 *
	 * * * * * * * * * * * * * * * * * * * * * * * * * * */

	void updateBoard(int[,] matrix)
	{
		var xLength = matrix.GetLength (1);
		var yLength = matrix.GetLength (0);
		int output;
		for (int yIndex = 0; yIndex < yLength; yIndex++)
		{
			for (int xIndex = 0; xIndex < xLength; xIndex++)
			{
				putPiece (xIndex, yIndex, matrix [yIndex, xIndex]);
			}
		}	
	}

	void resetMatrix()
	{
		boardMatrix = new int[,] {
			{1, -1, -1, 0, -1, -1, -1},
			{-1, -1, -1, -1, -1, -1, -1},
			{-1, -1, -1, -1, -1, -1, -1},
			{-1, -1, -1, -1, -1, -1, -1},
			{-1, -1, -1, -1, -1, -1, -1},
			{-1, -1, -1, -1, -1, -1, 0},
		};
	}

	//Prints the contents inside the matrix onto the console.

	void print2DMatrix(int[,] matrix)
	{
		string output = "";
		var xLength = matrix.GetLength (1);
		var yLength = matrix.GetLength (0);
		for (int yIndex = 0; yIndex < yLength; yIndex++)
		{
			for (int xIndex = 0; xIndex < xLength; xIndex++)
			{
				output = output + matrix[yIndex, xIndex] + " ";
			}
			output = output + "\n";
		}
		Debug.Log (output);
	}
}
