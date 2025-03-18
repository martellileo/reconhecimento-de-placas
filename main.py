import cv2
from ultralytics import YOLO
import pytesseract

# Carregar o modelo treinado
model = YOLO("ia\\model_- 16 march 2025 10_13.pt")  # Corrigindo o caminho

# Carregar a imagem de entrada
img = cv2.imread("ia\\images.jpg")

# Detectar as placas (detectando a região que contém a placa)
results = model(img)

# Se houver resultados, acessar o primeiro e renderizar
if results:
    result = results[0]  # Pega o primeiro resultado
    result.show()  # Exibe a imagem com detecções (substitui render())

    # Obter as coordenadas da caixa delimitadora da placa
    plates = result.boxes.xyxy.cpu().numpy()  # Resultado da detecção

    # Processar cada placa detectada
    for plate in plates:
        x1, y1, x2, y2 = map(int, plate[:4])  # Pegando coordenadas
        conf = plate[4] if len(plate) > 4 else 1.0  # Confiança da detecção

        if conf > 0.5:  # Se a confiança for maior que 50%, considere a detecção válida
            roi = img[y1:y2, x1:x2]  # Recortar a região da placa
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Processar a região da placa para OCR
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza
            _, thresh = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY)  # Aplicar limiarização
            text = pytesseract.image_to_string(thresh, config='--psm 8')  # Extração do texto

            print(f"Placa detectada: {text.strip()}")

# Mostrar a imagem com a caixa da placa
cv2.imshow("Deteccao de Placa", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
