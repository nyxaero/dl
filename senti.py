
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Bidirectional,Dense,Dropout

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb



num_words = 10000
maxlen = 200

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words = num_words)

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2,random_state=42)


model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=maxlen),
    Bidirectional(LSTM(64, return_sequences = False)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()
   
history = model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=5,  batch_size=64)

loss, accuracy = model.evaluate(x_test,y_test)


def decode_review(encoded_review):
    word_index = imdb.get_word_index()
    reverse_word_index = {value:key for key,value in word_index.items()}
    decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review if i > 2 ] )
    return decoded_review

def predict_sentiment(review):
    word_index = imdb.get_word_index()
    encoded_review = [word_index.get(word,0) + 3 for word in review.split()]
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)
    prediction = model.predict(padded_review, verbose=0)
    sentiment = 'positive' if prediction[0][0] >=0.5 else 'negative'
    return sentiment



for i in range(5):
    review = decode_review(x_test[i])
    sentiment = predict_sentiment(review)
    
    print(review, sentiment)