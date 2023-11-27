# Sentiment Analysis in Reviews in PT-BR

The dataset that was the base for this project can be found at: https://www.kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

## Why 
This project was made to learn more about NLP and sentiment analysis. Beyond that, I was hired to make this project for a company that wanted to know the sentiment reviews, and to have a higher accuracy, I made an ensemble.

## Requirements
Can be found at [requirements.txt](https://github.com/davirpp/Sentiment-Analysis-in-Reviews-in-PT-BR/blob/main/requirements.txt)

## Development
First of all, I searched about and use some architectures to make the sentiment analysis, like MLP, LSTM, GRU, CNN, and others. Then I make some variations of these architectures and trained one by one (the notebook of training is [training.ipynb](https://github.com/davirpp/Sentiment-Analysis-in-Reviews-in-PT-BR/blob/main/training.ipynb)).

Then, after training, I want to analyze the results of every model, so I made a notebook ([testing_models.ipynb](https://github.com/davirpp/Sentiment-Analysis-in-Reviews-in-PT-BR/blob/main/testing_models.ipynb)), there I get some reviews from some websites and different products and the whole dataset that I used to train the models(including train and test). After that analysis I choose the best models mixing the results of the models and the results of the analysis.

## Ensemble
The ensemble ([ensemble.py](https://github.com/davirpp/Sentiment-Analysis-in-Reviews-in-PT-BR/blob/main/ensemble.py)) was made with the best models that I choose. The models were:
- MLP
- 2x BiLSTM

The architecture of each model can be found at [testing_models.ipynb](https://github.com/davirpp/Sentiment-Analysis-in-Reviews-in-PT-BR/blob/main/testing_models.ipynb). The ensemble was made with the average of the results of the models.

The algorithm is simple:
- Get a raw string as input
- The string is processed and returned as embedding
- The embedding is passed to each model
- The results of each model are averaged
- The result is returned


## Challenges
The first version of the ensemble I used 7 models and it was pretty slow to make a single prediction (~10s). So I decided to use only 3 models and it was faster (~3s). But I wanted to make it even faster, because the commapny that hired me wanted to make this prediction to over 300 reviews at once. So after some studies I found a way using a decorator of TensorFlow that using this the final time to give the result for each prediction using CPU is ~0.08s. A huge difference that made me happy of the speed and accuracy of the model


Any doubts fell free to contact!
