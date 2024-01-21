import numpy as np

# Počet input nodes
i_nodes = 1

# Počet output nodes
o_nodes = 1

# Počet hidden nodes - střední vrstva neuronové sítě
h_nodes = 20

# Objem training dat - čim víc tím déle trvá učení ale tím víc toho bude síť umět
training_amount = 100

# Zavedení náhodných inputů pro učení
i_data = np.random.randn(training_amount, i_nodes)

# Ke každému číslu přiřazení jeho dvojnásobku (síť se naučí násobit inputy dvěma - jde ale zaměnit za jakékoli jiné číslo)
# Nepoužívám zde tzv. bias takže asi není možné ji naučit sčítat/odčítat a kvůli jen jedné hidden layer nemůže ani mocnit atd.
o_data = i_data * 2

# Zavedení náhodných weights pro obě vrstvy
w1 = np.random.randn(i_nodes, h_nodes)
w2 = np.random.randn(h_nodes, o_nodes)

# Loop učení - 100000 vzniklo pokus/omylem
for i in range(60000):
    h_values = i_data.dot(w1)  # Outputy po první vrstvě
    h_relu = np.maximum(h_values, 0)  # Aktivační funkce první vrstvy
    o_data_predictions = h_relu.dot(w2)  # Outputy druhé vrstvy (což je poslední vrstva, takže jde o finální predikce sítě z inputu)

    loss = np.square(o_data_predictions - o_data).sum()  # "Chyba" sítě
    print(loss)

    # Backpropagation/gradient descent - hlavní princip machine learning založení na derivacích
    grad_pred = 2 * (o_data_predictions - o_data)
    grad_w2 = h_relu.T.dot(grad_pred)
    grad_h_relu = grad_pred.dot(w2.T)
    grad_h_values = grad_h_relu
    grad_h_values[h_values < 0] = 0
    grad_w1 = i_data.T.dot(grad_h_values)

    # Doslova poučení z chyb a upravění weights hodnot
    w1 = w1 - grad_w1 * 1e-7
    w2 = w2 - grad_w2 * 1e-7


# Předpověd neuronové sítě na input
def predict(input_number):
    i_data = np.random.randn(1, 1)
    i_data[0][0] = input_number
    h_values = i_data.dot(w1)
    h_relu = np.maximum(h_values, 0)
    prediction = h_relu.dot(w2)
    return prediction


# Loop na vyzkoušení funkčnosti modelu - kvůli struktuře učení sítě musí byt input v 2D listu
while True:
    prediction = predict(input("Input number:"))
    print(f"Neural network output: {prediction}")
