Overview
This project implements a text classification model using TensorFlow and Keras. The data is stored in a Google Cloud Storage bucket and is fetched into a Google Colab environment for training. The model is trained on a dataset of sentences with multiple labels, using a Bidirectional LSTM architecture to predict the categories associated with the text.
Prerequisites
Before running the script, ensure you have the following installed:
Python 3.x
Google Colab
Required Python packages:
tensorflow
pandas
matplotlib
scikit-learn
google-cloud-storage
You can install these dependencies using the following command:
bash
Copy code
!pip install tensorflow pandas matplotlib scikit-learn

Authentication and Setup
This script fetches data from Google Cloud Storage, so authentication is required:
Make sure you have access to a Google Cloud project and have set up a storage bucket.
Authenticate the Colab environment with your Google Cloud account:
python
Copy code
from google.colab import auth
auth.authenticate_user()

Project Structure
The dataset used for training the model is stored in a Google Cloud Storage bucket named nathan-projects. The specific file being accessed is core/sectiontitles - TransformedData.csv. The script downloads this file from the bucket and loads it into a pandas DataFrame.
File Paths
The file structure in Google Cloud Storage is defined using the os.path.join function. Here's an example of the file path being used:
python
Copy code
file_name = os.path.join('core', 'sectiontitles - TransformedData.csv')

Ensure that you have the correct file path and bucket name to access your data.
Data Loading
The dataset is loaded into a pandas DataFrame after being fetched from the cloud. You can inspect the first few rows of the dataset using:
python
Copy code
train_df.head()

The column Sentence represents the input features (sentences), and columns from index 3 onwards represent the target labels.
Model Architecture
The model is built using TensorFlow's Keras API. The architecture includes:
Text Vectorization:
The sentences are tokenized and vectorized using the TextVectorization layer, which converts each word to an integer representation. The maximum vocabulary size is set to 10000, and each sequence is capped at 180 tokens.
Bidirectional LSTM:
The core of the model is a Bidirectional LSTM, which allows the model to learn from both past and future words in a sentence.
Fully Connected Layers:
The model has multiple dense layers for feature extraction with ReLU activation functions.
Output Layer:
The final layer has 16 units with a softmax activation function, suitable for multi-class classification.
Model Training
The dataset is split into three parts:
Training Set: 60% of the dataset
Validation Set: 20% of the dataset
Test Set: 20% of the dataset
Data is preprocessed using TensorFlow's data pipeline methods like caching, shuffling, batching, and prefetching.
The model is compiled with the following parameters:
Loss: categorical_crossentropy
Optimizer: Adam
Metrics: accuracy
The model is trained for 10 epochs:
python
Copy code
history = model.fit(train, epochs=10, validation_data=val)

Output
The model will output the training and validation accuracy and loss for each epoch, which can be visualized later using matplotlib.
Notes
You can adjust the number of epochs, batch size, or model architecture depending on your dataset size and complexity.
Ensure you have a valid Google Cloud project and the necessary permissions to access your storage bucket.
