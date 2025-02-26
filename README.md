# TFM_PROYECTO
Proyecto para Trabajo de Fin de Máster de Inteligencia Artificial
"# TFM_PROYECTO_UNIR_2024" 

# Comando para crear entorno virtual
python -m venv .tfm-venv

# Comando para activar entorno virtual en Windows
.tfm-venv\Scripts\activate

# Comando para activar entorno virtual en Linux
source .tfm-venv/bin/activate

# Requirements (paquetes necesarios para correr el proyecto)
pip install -r requirements.txt

# Ejecutar modelos CNN y VIT
python train.py

# Levantar sevidor Django
python manage.py runserver

# Desactivar máquina virtual
deactivate
