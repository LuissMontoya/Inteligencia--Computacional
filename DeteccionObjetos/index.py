import cv2
import numpy as np

# Cargar YOLOv3
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Obtener los nombres de las capas de salida
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Cargar la imagen donde se desea detectar objetos
img = cv2.imread("images/img0.jpg")

if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta y el nombre del archivo.")
    exit()

# Obtener las dimensiones de la imagen
height, width, channels = img.shape

# Preprocesar la imagen para YOLOv3
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Información sobre las detecciones
class_ids = []
confidences = []
boxes = []

# Procesar cada una de las detecciones
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:  # Umbral de confianza
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Aplicar Non-Maximum Suppression para eliminar cuadros duplicados
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Dibujar los cuadros delimitadores alrededor de los objetos detectados
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(class_ids[i])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Mostrar la imagen con los objetos detectados
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
