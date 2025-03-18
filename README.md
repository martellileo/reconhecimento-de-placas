1. pip install torch opencv-python ultralytics

2. pip install pytesseract

3. 🔹 Como resolver?
1️⃣ Instalar o Tesseract OCR
Se você ainda não instalou, baixe e instale o Tesseract OCR para Windows:

🔗 Download: https://github.com/UB-Mannheim/tesseract/wiki
Baixe o instalador mais recente para Windows e instale.

2️⃣ Adicionar o Tesseract ao PATH
Após instalar, você precisa adicionar o caminho do Tesseract ao PATH do Windows.

C:\Program Files\Tesseract-OCR

Adicione ao PATH
No Windows, abra Configurações do Sistema > Variáveis de Ambiente.
Em Variáveis do Sistema, encontre Path, edite e adicione.

Abra o Prompt de Comando (cmd) e digite:
tesseract --version

Se ele retornar a versão do Tesseract, está funcionando!