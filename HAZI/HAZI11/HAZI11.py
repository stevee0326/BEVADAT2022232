import tensorflow as tf

'''
Készíts egy metódust ami a cifar100 adatbázisból betölti a train és test adatokat. (tf.keras.datasets.cifar100.load_data())
Majd a tanitó, és tesztelő adatokat normalizálja, és vissza is tér velük.


Egy példa a kimenetre: train_images, train_labels, test_images, test_labels
függvény neve: cifar100_data
'''
def cifar100_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    return train_images, train_labels, test_images, test_labels


'''
Készíts egy konvolúciós neurális hálót, ami képes felismerni a képen mi van a 100 osztály közül.
A háló kimenete legyen 100 elemű, és a softmax aktivációs függvényt használja.
Hálon belül tetszőleges számú réteg lehet..


Egy példa a kimenetre: model,
return type: keras.engine.sequential.Sequential
függvény neve: cifar100_model
'''
def cifar100_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(100, activation='softmax')
    ])
    return model



'''
Készíts egy metódust, ami a bemeneti hálot compile-olja.
Optimizer: Adam
Loss: SparseCategoricalCrossentropy(from_logits=False)

Egy példa a bemenetre: model
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_compile
'''
def model_compile(model):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model



'''
Készíts egy metódust, ami a bemeneti hálót feltanítja.

Egy példa a bemenetre: model,epochs, train_images, train_labelsz
Egy példa a kimenetre: model
return type: keras.engine.sequential.Sequential
függvény neve: model_fit
'''
def model_fit(model, epochs, train_images, train_labels):
    model.fit(train_images, train_labels, epochs=epochs)
    return model

'''
Készíts egy metódust, ami a bemeneti hálót kiértékeli a teszt adatokon.

Egy példa a bemenetre: model, test_images, test_labels
Egy példa a kimenetre: test_loss, test_acc
return type: float, float
függvény neve: model_evaluate
'''
def model_evaluate(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    return test_loss, test_acc





