# Neural_Network_Model
An artificial neural network classifier was built to predict a student chance of getting accepted in graduate school using three features (GRE_score, TOEFL _score, and GPA).

Training Set:

GRE Score	  TOEFL 	GPA 	Chance of Acceptance
300	        99	    6.8	    0.36
308	        103	    8.36	  0.7
329	        110	    9.15	  0.84
332	        118	    9.36	  0.9

Testing Set:

GRE Score	  TOEFL	  GPA	  Chance of Acceptance
296	        95	    7.54	   0.44
293	        97	    7.8	     0.64
325	        112	    8.96	   0.8



Steps:

1.	Build a two-layers neural network of one input layer (three nodes x_1, x_2 and b or θ_0), one hidden layer (three nodes with sigmoid activation function) and one output layer (one node with no activation function). 
2.	Initialize the parameters with some random values.
3.	Perform a feed-forward stage to propagate the input forward through the network. 
4.	Preform the back-propagation algorithm using sum square error (SSE).
5.	Print the content of the input/output of hidden layers and output layer at the feed forward stages.
6.	Print the content of the weight matrices after each epoch (4 epochs).
7.	Test the result of classifying the testing examples.
