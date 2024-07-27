# Stock-Prediction-Model

This project predicts a company's stock prices using a Recurrent Neural Network (RNN). It trains the model on historical stock prices, leveraging TensorFlow's Keras API. The model consists of a SimpleRNN layer with 100 units, a Dropout layer to prevent overfitting, a GRU layer with 50 units, and a Dense layer for output. The data preprocessing includes scaling prices using `StandardScaler` and creating sequences of 50 previous prices as input (`x_train`) with the corresponding next price as output (`y_train`). The model is trained with the Adam optimizer and Mean Squared Error (MSE) loss function, utilizing early stopping to improve training efficiency.

To use this project, load and preprocess the dataset, create the training and testing datasets, and build and train the RNN model. Once trained, the model predicts stock prices on the test set, and the performance is evaluated using the Root Mean Square Error (RMSE). The RMSE score helps understand the model's accuracy in forecasting stock prices. 

