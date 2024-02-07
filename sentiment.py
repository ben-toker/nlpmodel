from fastai.learner import *

# Load the model
learn = load_learner("Models\\Sentiment_Model_2.2")

def sentiment_predict(text):
    prediction = learn.predict(text)

    if prediction[0] == 'pos':
        return "Negative"
    else:
        return "Positive"


