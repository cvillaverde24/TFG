# TFG
Repositorio para el Trabajo Fin de Grado del doble grado en ingeniería informática y matemáticas: Detección y conteo de baobab en imágenes de Google Earth usando Deep Learning.

En este repositorio encontramos el código y las directrices necesarias para entrenar un modelo basado en la librería Detectron2 para la detección de baobabs en imágenes de Google Earth.
Encontramos el fichero "tococo.py" que ha sido el utilizado para pasar la notación de las imágenes etiquetadas de formato XML a COCO format.

También tenemos el fichero train.py que es el código principal del proyecto, pero antes de ejecutarlo necesitamos descargarnos las imágenes que vamos a utilizar.
Estas imágenes se encuentran en el siguiente enlace: https://drive.google.com/drive/folders/1chJ7-vvkqYPew3F9RyicUtuk2Ni99pj1

Aquí encontraremos dos carpetas: baobabs_dataset_con_sombra y baobabs_dataset_sin_sombra, las cuales deberemos de tener en la misma carpeta que el fichero train.py
Dependiendo de como queramos entrenar el modelo (con sombra o sin sombra) debemos de renombrar la carpeta correspondiente a "baobabs_dataset".
Debemos crear también una carpeta llamada "output_images", que será en la que se guarden los resultados del test, es decir, las imágenes con los baobabs detectados en el test.
Una vez tengamos esto podemos ejecutar el fichero con: python train.py

Como salida tendremos las imágenes del test y se creará una carpeta "output" en la que podremos ver las métricas obtenidas.

Debemos tener en cuenta que para poder ejecutar el fichero debemos de tener una tarjeta gráfica CUDA compatible o ejecutar el fichero en los servidores de la universidad en los cuales deberemos crear un entorno virtual, tanto como si lo ejecutamos en nuestro equipo como si lo hacemos en los servidores, necesitamos instalar la librería Detectron2 y para ello es necesario tener como sistema operativo Linux o macOS con python con una versión igual o superior a 3.7, tener instalado Pytorch con versión igual o superior a 1.8, torchvision y OpenCV (Ver https://detectron2.readthedocs.io/en/latest/tutorials/install.html para más información).
