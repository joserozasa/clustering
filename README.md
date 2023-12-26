# catalogo_algas

#crop_images = Funcion para crear los recortes a partir de una imagen

#clustering = Funcion para agrupar los recortes

# Versión con GPU

En un entorno viurtual con python 3.9 ejecutar los siguientes comandos para instalar el detector agnóstico.

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

mkdir retina_folder

cd retina_folder/

git clone https://github.com/amirbar/DETReg

cd DETReg/

git fetch && git checkout feature/demo

pip install -r requirements.txt

cd models/ops/

sudo apt-get install nvidia-modprobe

sh ./make.sh

cd ..

Se entregan dos archivos .py con sus respectivos main de prueba

# Versión con CPU (sin GPU)

## Instalación:

1. Descargar Docker Desktop desde https://www.docker.com/. Instalar y ejecutar.
2. Abrir una terminal. En Windows, para esto se ejecuta "cmd" sin las comillas.
3. Ir al directorio donde se haya clonado este directorio usando el comando "cd". Ej: si el repositorio se clonó en C:\Documentos\catalogo_algas, ejecutar:
    cd C:\Documentos\catalogo_algas
4. Ejecutar la siguiente línea para construir el contenedor Docker:
    docker build -t <nombre_de_docker> .
    Donde <nombre_de_docker> es el nombre que le darán al Docker. El comando anterior termina con un punto. Es muy importante que esté, ya que simboliza el directorio del repositorio clonado que se transferirá dentro del Docker.
    Ej: si el Docker se llamará "catalga" al Docker, entonces ejecutar:
    docker build -t catalga .
    OJO: Este paso puede tardar bastante, pero se ejecutará solamente cuando sea necesario construir el Docker.

## Funcionamiento
1. Ejecutar en el terminal:
docker run --rm -v <directorio_local>:/files/ <nombre_de_docker> -d /files/<subdirectorio_de_imagenes> -p {gpu,cpu} -minclu <numero_minimo_de_clusters> -maxclu <numero_minimo_de_clusters>

Donde:
 <directorio_local>: directorio que se compartirá con el contenedor Docker. En este directorio o en algún subdirectorio estarán las imágenes a analizar. Dentro del contenedor, este directorio se llamará /files/
 <subdirectorio_de_imagenes>: subdirectorio relativo en que están las imágenes.
 -p: parámetro de procesador. Las opciones son: gpu y cpu (utilizar esta opción al no tener tarjeta gráfica NVIDIA)
 -minclu: número mínimo de clusters. Si no se especifica, la cantidad por defecto es 10.
 -maxclu: número mínimo de clusters. Si no se especifica, la cantidad por defecto es 30.

Ej. 1:
- Se quiere compartir todo el directorio de documentos C:\Documents y las imágenes están en un subdirectorio llamado "microalgas".
- El equipo no tiene GPU NVIDIA.
- Se busca probar con una cantidad de cluster entre 3 y 5 para encontrar el mejor número.
- El nombre dado al Docker durante la instalación es "catalga".
Entonces el comando a ejecutar es:
docker run --rm -v "C:\Documents":/files/ catalga -d /files/microalgas -p cpu -minclu 3 -maxclu 5

Ej. 2:
- En vez de compartir todo el directorio de documentos con el contenedor Docker, se comparte directamente el directorio de imágenes C:\Documents\microalgas
- Se utilizarán las cantidades mínima y máxima de clusters por defecto.
- Todo lo demás se mantiene.
Entonces el comando a ejecutar es:
docker run --rm -v "C:\Documents\microalgas":/files/ catalga -d /files/ -p cpu

2. Una vez terminada la ejecución, se crearán dos subdirectorios dentro del directorio de imágenes:
- crops: recortes obtenidos a partir de las detecciones. El formato de nombre es: <nombre_de_imagen>_<xtop>_<yleft>_<xbottom>_<yright>.jpg
- catalog: recortes de "crops" agrupados. En su interior, por cada ejecución se crea una carpeta con el nombre del timestamp según el horario UTC. Dentro de estas carpetas encontraremos subcarpetas con nombres entre 0 y N-1, donde N es el número óptimo de clusteres encontrado.
