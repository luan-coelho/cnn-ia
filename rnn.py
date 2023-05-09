import json

import numpy as np
from keras import layers, models


def create_lstm_model():
    model = models.Sequential()
    model.add(layers.Embedding(input_dim=10000, output_dim=32))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Carregar e preprocessar seus dados de treino aqui
train_data, train_labels = np.random.randint(0, 10000, (1000, 100)), np.random.randint(0, 2, (1000,))

lstm_model = create_lstm_model()
lstm_model.fit(train_data, train_labels, epochs=10, batch_size=32)


def load_config():
    with open("config.json", "r") as file:
        return json.load(file)


config = load_config()


def preprocess_request(request):
    # Extrair a URL da requisição
    request_url = request.path

    # Codificar a URL como uma sequência de inteiros
    encoded_url = [ord(c) for c in request_url]

    # Limitar o comprimento da sequência e preencher com zeros, se necessário
    max_len = 100
    padded_url = np.zeros(max_len, dtype=np.int32)
    padded_url[:len(encoded_url)] = encoded_url[:max_len]

    # Redimensionar a entrada para ser compatível com a LSTM (1, 100)
    lstm_input = np.expand_dims(padded_url, axis=0)

    return lstm_input


def is_suspicious_request(request):
    # Verifique se a URL está na lista de URLs suspeitas
    if request.path in config["suspicious_urls"]:
        return True

    # Pré-processar a requisição e obter a entrada compatível com a LSTM
    lstm_input = preprocess_request(request)

    # Use a LSTM para classificar a requisição como normal (0) ou suspeita (1)
    prediction = lstm_model.predict(lstm_input)
    predicted_class = np.round(prediction[0][0])

    return predicted_class == 0
