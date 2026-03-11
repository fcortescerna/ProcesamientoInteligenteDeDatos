import numpy as np
import tensorflow as tf
from urllib import parse
from http.server import HTTPServer, BaseHTTPRequestHandler

# El modelo fue guardado con tf.nn.softmax (softmax_v2) en una version antigua de TF.
# Se usa custom_objects para que la version nueva de Keras lo reconozca correctamente.
model = tf.keras.models.load_model(
    'modelo_mnist.keras',
    custom_objects={'softmax_v2': tf.keras.activations.softmax}
)
print("Modelo cargado correctamente.")
class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        print("Peticion recibida")
        #Obtener datos de la petición y limpiar los datos
        content_length = int(self.headers['Content-Length'])
        data = self.rfile.read(content_length)
        data = data.decode().replace('pixeles=', '')
        data = parse.unquote(data)
        #Realizar transformación para dejar igual que los ejemplos que usa MNIST
        arr = np.fromstring(data, np.float32, sep=",")
        arr = arr.reshape(28,28)
        arr = np.array(arr)
        arr = arr.reshape(1,28,28,1)
        #Realizar y obtener la predicción
        prediction_values = model.predict(arr, batch_size=1)
        prediction = str(np.argmax(prediction_values))
        print("Prediccion final: " + prediction)
        #Regresar respuesta a la peticion HTTP
        self.send_response(200)
        #Evitar problemas con CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(prediction.encode())

#Iniciar el servidor en el puerto 8000 y escuchar por siempre
#Si se queda colgado, en el admon de tareas buscar la tarea de python y finalizar tarea
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()