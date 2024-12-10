from torch import nn
from torch.utils.data import DataLoader
import torch
from torchvision import datasets # Para descargar el dataset
from torchvision.transforms import ToTensor # Para convertir los datos a Tensores
import matplotlib.pyplot as plt # Para graficar las imágenes + categorías

# Source: https://www.youtube.com/watch?v=QetoD5LXlEg&ab_channel=CodificandoBits


# Crear la Red Neuronal como una subclase de nn.Module
# Siempre se añaden dos métodos a esta subclase
# 1. Método "init": define la arquitectura de la red
# 2. Método "forward": define cómo será generada cada predicción

class RedNeuronal(nn.Module):
    # 1. Método "init"
    def __init__(self):
        super().__init__()

        # Y agregar secuencialmente las capas
        self.aplanar = nn.Flatten() # Aplanar imágenes de entrada
        self.red = nn.Sequential(
            nn.Linear(28*28, 15), # Capa de entrada + capa oculta
            nn.ReLU(), # Función de activación capa oculta
            nn.Linear(15, 10), # Capa de salida SIN activación
        )

    # 2. Método "forward" (x = dato de entrada)
    def forward(self, x):
        # Definir secuencialmente las operaciones a aplicar
        x = self.aplanar(x) # Aplanar dato
        logits = self.red(x) # Generar predicción.
                             # Logits es la salida sin normalizar

        return logits


# Detectar la GPU
device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Usando {device}")

# Instancia (llevada a la GPU)
modelo = RedNeuronal().to(device)

# Crear el directorio "datos" y ejecutar el siguiente código:
data_mnist = datasets.MNIST(
    root = "datos", # Carpeta donde se almacenará
    train=True, # True: 60.000 imágenes, False: 10.000 imágenes
    download=True,
    transform=ToTensor() # Convertir imágenes a tensores
)

torch.manual_seed(123)
train, val, test = torch.utils.data.random_split(
    data_mnist, [0.8, 0.1, 0.1]
)

# Verificar tamaños
print(f'Tamaño set de entrenamiento: {len(train)}')
print(f'Tamaño set de validación: {len(val)}')
print(f'Tamaño set de prueba: {len(test)}')

# Y verificar el tipo de dato de train, val y test
print(f'Tipo de dato set "train": {type(train)}')
print(f'Tipo de dato set "val": {type(val)}')
print(f'Tipo de dato set "test": {type(test)}')



# Definir el tamaño del lote
TAM_LOTE = 1000 # batch size

# Crear los "dataloaders" para los sets de entrenamiento y validación
train_loader = DataLoader(
    dataset=train,
    batch_size=TAM_LOTE,
    shuffle=True # Mezclar los datos aleatoriamente al crear cada lote
)

val_loader = DataLoader(
    dataset=val,
    batch_size=TAM_LOTE,
    shuffle=False
)


# Hiperparámetros
TASA_APRENDIZAJE = 0.1 # learning rate (0.1)
EPOCHS = 10 # Número de iteraciones de entrenamiento

# Función de pérdida y optimizador
fn_perdida = nn.CrossEntropyLoss()
optimizador = torch.optim.SGD(modelo.parameters(), lr=TASA_APRENDIZAJE)


def train_loop(dataloader, model, loss_fn, optimizer):
    # Cantidad de datos de entrenamiento y cantidad de lotes
    train_size = len(dataloader.dataset)
    nlotes = len(dataloader)

    # Indicarle a Pytorch que entrenaremos el modelo
    # No entrena como tal el modelo, simplemente indica
    # que será entrenado (cuando es validado, se usa eval())
    model.train()

    # Inicializar acumuladores pérdida y exactitud
    perdida_train, exactitud = 0, 0

    # Presentar los datos al modelo por lotes (de tamaño TAM_LOTE)
    for nlote, (X, y) in enumerate(dataloader):
        # Mover "X" y "y" a la GPU
        X, y = X.to(device), y.to(device)

        # Forward propagation
        logits = model(X)

        # Backpropagation
        loss = loss_fn(logits, y) # 'logits' es la prediccion y 'y' es el valor real
        loss.backward()  # calcula los gradientes
        optimizer.step() # actualiza los parametros (pesos)
        optimizer.zero_grad() # resetea los gradientes

        # Acumular valores de pérdida y exactitud
        # perdida_train <- perdida_train + perdida_actual
        # exactitud <- exactitud + numero_aciertos_actuales
        perdida_train += loss.item()
        exactitud += (logits.argmax(1)==y).type(torch.float).sum().item()

        # Imprimir en pantalla la evolución del entrenamiento (cada 10 lotes)
        if nlote % 10 == 0:
            # Obtener el valor de la pérdida (loss) y el número de datos procesados (ndatos)
            ndatos = nlote*TAM_LOTE

            # E imprimir en pantalla
            print(f"\tPérdida: {loss.item():>7f}  [{ndatos:>5d}/{train_size:>5d}]")

    # Al terminar de presentar todos los datos al modelo, promediar pérdida y exactitud
    perdida_train /= nlotes # Pérdida promedio = pérdida acumulada / número de lotes
    exactitud /= train_size # Exactitud promedio = exactitud acumulada / número de datos

    # E imprimir información
    print(f'\tExactitud/pérdida promedio:')
    print(f'\t\tEntrenamiento: {(100*exactitud):>0.1f}% / {perdida_train:>8f}')



def val_loop(dataloader, model, loss_fn):
    # Cantidad de datos de validación y cantidad de lotes
    val_size = len(dataloader.dataset)
    nlotes = len(dataloader)

    # Indicarle a Pytorch que validaremos el modelo
    model.eval()

    # Inicializar acumuladores pérdida y exactitud
    perdida_val, exactitud = 0, 0

    # Evaluar (generar predicciones) usando "no_grad"
    # no necesitamos calcular los gradientes porque no haremos
    # backprop
    with torch.no_grad():
        for X, y in dataloader:
            # Mover "X" y "y" a la GPU
            X, y = X.to(device), y.to(device)

            # Propagación hacia adelante (predicciones)
            logits = model(X)

            # Acumular valores de pérdida y exactitud
            perdida_val += loss_fn(logits, y).item()
            exactitud += (logits.argmax(1) == y).type(torch.float).sum().item()

    # Tras generar las predicciones calcular promedios de pérdida y exactitud
    perdida_val /= nlotes
    exactitud /= val_size

    # E imprimir en pantalla
    print(f"\t\tValidación: {(100*exactitud):>0.1f}% / {perdida_val:>8f} \n")


for t in range(EPOCHS):
    print(f"Iteración {t+1}/{EPOCHS}\n-------------------------------")
    # Entrenar
    train_loop(train_loader, modelo, fn_perdida, optimizador)
    # Validar
    val_loop(val_loader, modelo, fn_perdida)
    print("Listo, el modelo ha sido entrenado!")


def predecir(model, img):
    # Generar predicción
    logits = model(img)
    y_pred = logits.argmax(1).item()

    # Mostrar imagen original y categoría predicha
    plt.imshow(img.cpu().squeeze(), cmap="gray")
    plt.title(f'Categoría predicha: {y_pred}');


# Tomar una imagen del set de prueba
img, lbl = test[1235]

# Y generar la predicción
predecir(modelo, img)

