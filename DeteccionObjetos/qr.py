import cv2
import numpy as np

# Cargar la imagen
img = cv2.imread("Qrs/5.jpg")

# Verificar si la imagen se ha cargado correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el nombre del archivo.")
    exit()

# Obtener las dimensiones de la imagen
height, width, channels = img.shape

# Cargar YOLOv3-tiny
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Obtener los nombres de las capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Preprocesar la imagen
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Procesar las detecciones
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        for obj in detection:
            obj = obj.flatten()  # Asegúrate de que obj sea un array 1D
            if obj.size == 85:  # Verifica el tamaño correcto para YOLOv3-tiny
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                confidence = obj[4]  # La confianza está en la posición 4
                scores = obj[5:]  # Las puntuaciones de clase están después de la posición 4
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.7:  # Umbral de confianza ajustado
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

# Aplicar Non-Maximum Suppression
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3)  # Ajustar umbrales de NMS

# Dibujar los cuadros delimitadores
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "QR Code", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Mostrar la imagen
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
