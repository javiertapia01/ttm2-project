# PyTorch básico I  
## Creación de tensores en PyTorch

**Proyecto:** Proyecto final MAT2320 — TTM2  
**Carril:** B — Alfabetización técnica mínima orientada a TTM2  
**Semana:** 2  
**Sesión:** B  
**Estado:** Versión inicial del apunte técnico  
**Autor:** Javier Tapia  
**Contexto:** Proyecto MAT2320 / TTM2 / preparación técnica para lectura e implementación en PyTorch-DGL  

---

## 1. Resumen

Este apunte documenta el avance inicial de la sesión B de la semana 2 del proyecto TTM2, correspondiente al Carril B: alfabetización técnica mínima orientada al paper y a su futura implementación.

El foco de esta versión es la **creación de tensores en PyTorch**, entendida como el primer paso práctico para adquirir fluidez en el lenguaje tensorial usado en Deep Learning y, posteriormente, en el código asociado a arquitecturas geométricas equivariantes.

La sesión no busca estudiar PyTorch de manera abstracta o enciclopédica. Su objetivo es más concreto: construir una base operativa que permita leer código, ejecutar micro-pruebas y comenzar a interpretar cómo las estructuras matemáticas del paper se representan computacionalmente.

---

## 2. Objetivo del apunte

El objetivo de este documento es fijar, de manera clara y reproducible, los conceptos mínimos necesarios para crear tensores en PyTorch.

Al finalizar esta parte, se espera poder:

1. explicar qué es PyTorch a nivel básico;
2. entender por qué PyTorch es importante en Deep Learning;
3. distinguir un tensor de PyTorch de una lista ordinaria de Python;
4. crear tensores desde datos explícitos;
5. crear tensores con forma prescrita;
6. generar tensores mediante secuencias y valores aleatorios;
7. inspeccionar propiedades básicas como `shape` y `dtype`;
8. preparar micro-ejemplos que puedan usarse en el repositorio del proyecto.

---

## 3. Contexto dentro del proyecto TTM2

TTM2 es un proyecto teórico-práctico centrado en el estudio de arquitecturas geométricas equivariantes, en particular modelos que combinan ideas de Geometric Deep Learning, tensores esféricos, representaciones de grupos, kernels equivariantes y mecanismos de atención.

Dado que el paper y su implementación usan PyTorch y DGL, resulta necesario adquirir una alfabetización técnica progresiva. Esta alfabetización no se plantea como un curso general de programación, sino como un aprendizaje just-in-time, guiado por las necesidades del paper.

En este contexto, aprender a crear tensores en PyTorch es el primer paso técnico natural, porque en Deep Learning casi todos los objetos computacionales relevantes se representan como tensores: datos, features, pesos, embeddings, activaciones, kernels y salidas intermedias.

---

## 4. ¿Qué es PyTorch?

PyTorch es una librería de Python orientada al cálculo tensorial y a la construcción de modelos de Deep Learning.

En términos prácticos, PyTorch permite:

- representar datos como arreglos multidimensionales;
- operar algebraicamente sobre esos arreglos;
- construir redes neuronales;
- calcular gradientes automáticamente;
- entrenar modelos mediante optimización numérica.

En esta etapa del proyecto no se estudiará todavía todo PyTorch. El objetivo inicial es dominar el uso básico de tensores, porque ellos son la unidad computacional fundamental sobre la cual se construyen los modelos.

---

## 5. ¿Por qué PyTorch es importante en Deep Learning?

En Deep Learning, los modelos se implementan mediante operaciones sobre tensores.

Por ejemplo:

- una imagen puede representarse como un tensor;
- un lote de datos puede representarse como un tensor;
- los pesos de una red neuronal son tensores;
- las activaciones intermedias de una red son tensores;
- los embeddings de nodos, aristas o tokens son tensores;
- los kernels aprendibles de una arquitectura también se almacenan como tensores.

Por tanto, aprender PyTorch implica aprender el lenguaje computacional en el que se expresan muchas arquitecturas modernas de Deep Learning.

Para este proyecto, PyTorch es especialmente importante porque permitirá pasar progresivamente desde la formulación matemática de TTM2 hacia su representación computacional.

---

## 6. Tensor en PyTorch: definición operativa

En esta etapa inicial, se usará la siguiente definición operativa.

**Definición.** Un tensor en PyTorch es un arreglo multidimensional de números.

Según su número de dimensiones, podemos pensar en los siguientes casos básicos:

| Objeto intuitivo | Interpretación computacional |
|---|---|
| Escalar | Tensor de dimensión 0 |
| Vector | Tensor de dimensión 1 |
| Matriz | Tensor de dimensión 2 |
| Pila de matrices | Tensor de dimensión 3 |
| Arreglo multidimensional general | Tensor de dimensión mayor |

Esta definición es suficiente para comenzar a trabajar técnicamente con PyTorch.

---

## 7. Advertencia conceptual: tensor de PyTorch vs tensor geométrico

Es importante distinguir dos usos de la palabra “tensor”.

En PyTorch, un tensor es principalmente una estructura de datos: un arreglo multidimensional de números.

En cambio, en el paper TTM2 aparecen tensores en un sentido geométrico y representacional. Por ejemplo, los tensores esféricos son objetos que transforman bajo representaciones irreducibles de SO(3).

Por tanto:

- **tensor en PyTorch**: estructura computacional;
- **tensor geométrico/esférico**: objeto matemático con reglas de transformación bajo simetrías.

Ambos sentidos estarán relacionados más adelante, porque los tensores geométricos deberán representarse computacionalmente mediante tensores de PyTorch. Sin embargo, en esta primera etapa conviene no confundirlos.

---

## 8. Creación de tensores: idea general

Crear un tensor significa construir un objeto de PyTorch que contiene datos numéricos organizados con una forma determinada.

Toda creación de tensores responde a dos preguntas:

1. **¿Qué contenido tendrá el tensor?**
   - datos escritos a mano;
   - ceros;
   - unos;
   - un valor fijo;
   - una secuencia;
   - valores aleatorios.

2. **¿Qué forma tendrá el tensor?**
   - escalar;
   - vector;
   - matriz;
   - tensor de tres o más dimensiones.

La intuición básica es:

> crear un tensor = fijar contenido + fijar forma.

---

## 9. Importación de PyTorch

Antes de usar PyTorch, se importa la librería:

```python
import torch
```

Por convención, PyTorch se importa como `torch`.

Esta línea debe aparecer al inicio de los scripts que utilicen tensores de PyTorch.

---

## 10. Crear tensores desde datos explícitos

La forma más directa de crear un tensor es usar:

```python
torch.tensor(...)
```

Esta función convierte datos escritos en Python en un tensor de PyTorch.

---

### 10.1. Escalar

```python
import torch

a = torch.tensor(5)

print(a)
print(a.shape)
print(a.dtype)
```

**Interpretación.**

Este tensor contiene un único número. Es un tensor de dimensión 0.

Su `shape` será vacío, porque no tiene ejes ni dimensiones extendidas.

---

### 10.2. Vector

```python
b = torch.tensor([1, 2, 3])

print(b)
print(b.shape)
print(b.dtype)
```

**Interpretación.**

Este tensor tiene una dimensión. Puede interpretarse como un vector de largo 3.

Su forma esperada es:

```python
torch.Size([3])
```

---

### 10.3. Matriz

```python
c = torch.tensor([
    [1, 2],
    [3, 4]
])

print(c)
print(c.shape)
print(c.dtype)
```

**Interpretación.**

Este tensor tiene dos dimensiones. Puede interpretarse como una matriz de 2 filas y 2 columnas.

Su forma esperada es:

```python
torch.Size([2, 2])
```

---

### 10.4. Tensor tridimensional

```python
d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print(d)
print(d.shape)
print(d.dtype)
```

**Interpretación.**

Este tensor puede pensarse como una pila de dos matrices, cada una de tamaño 2 x 2.

Su forma esperada es:

```python
torch.Size([2, 2, 2])
```

---

## 11. Regularidad de la estructura

Al crear tensores desde listas, la estructura debe ser regular.

Por ejemplo, esto es válido:

```python
x = torch.tensor([
    [1, 2],
    [3, 4]
])
```

porque ambas filas tienen la misma longitud.

En cambio, esto es problemático:

```python
x = torch.tensor([
    [1, 2],
    [3]
])
```

La razón es que no define una matriz rectangular. Una fila tiene largo 2 y la otra largo 1.

**Principio práctico.**

Un tensor debe tener una forma bien definida. No puede ser una tabla irregular.

---

## 12. Crear tensores llenos de ceros: `torch.zeros`

La función `torch.zeros` crea un tensor lleno de ceros.

### Ejemplo: vector de ceros

```python
x = torch.zeros(3)

print(x)
print(x.shape)
```

Resultado esperado conceptualmente:

```python
tensor([0., 0., 0.])
torch.Size([3])
```

### Ejemplo: matriz de ceros

```python
y = torch.zeros(2, 4)

print(y)
print(y.shape)
```

Este tensor tiene forma:

```python
torch.Size([2, 4])
```

**Uso práctico.**

Los tensores de ceros se usan con frecuencia para inicializar estructuras, reservar espacio o construir ejemplos simples.

---

## 13. Crear tensores llenos de unos: `torch.ones`

La función `torch.ones` crea un tensor lleno de unos.

```python
x = torch.ones(5)

print(x)
print(x.shape)
```

También se puede crear una matriz:

```python
y = torch.ones(2, 3)

print(y)
print(y.shape)
```

**Uso práctico.**

Los tensores de unos sirven para construir ejemplos controlados, probar multiplicaciones, crear factores constantes o inicializar estructuras simples.

---

## 14. Crear tensores con un valor fijo: `torch.full`

La función `torch.full` crea un tensor de una forma dada, lleno con un valor específico.

```python
x = torch.full((2, 3), 7)

print(x)
print(x.shape)
```

**Interpretación.**

Este código crea una matriz de 2 filas y 3 columnas, donde cada entrada vale 7.

La forma se entrega como una tupla:

```python
(2, 3)
```

**Uso práctico.**

`torch.full` es útil cuando se quiere construir rápidamente un tensor con una constante distinta de 0 o 1.

---

## 15. Crear secuencias con `torch.arange`

La función `torch.arange` crea una secuencia de valores igualmente espaciados.

```python
x = torch.arange(5)

print(x)
print(x.shape)
```

Este código produce:

```python
tensor([0, 1, 2, 3, 4])
```

También puede usarse con inicio y fin:

```python
y = torch.arange(2, 8)

print(y)
```

Resultado conceptual:

```python
tensor([2, 3, 4, 5, 6, 7])
```

**Interpretación.**

`torch.arange` es análogo a `range` en Python, pero devuelve un tensor.

**Uso práctico.**

Es especialmente útil para construir ejemplos, índices o datos sintéticos simples.

---

## 16. Crear secuencias con `torch.linspace`

La función `torch.linspace` crea un número fijo de puntos entre dos extremos.

```python
x = torch.linspace(0, 1, steps=5)

print(x)
print(x.shape)
```

Resultado conceptual:

```python
tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
```

**Diferencia con `arange`.**

* `arange` se usa cuando se quiere avanzar por pasos.
* `linspace` se usa cuando se quiere una cantidad fija de puntos entre dos extremos.

---

## 17. Crear tensores aleatorios con `torch.rand`

La función `torch.rand` crea tensores con valores aleatorios uniformes entre 0 y 1.

```python
x = torch.rand(3)

print(x)
print(x.shape)
```

También puede crearse una matriz:

```python
y = torch.rand(2, 2)

print(y)
print(y.shape)
```

**Interpretación.**

Los valores serán distintos en cada ejecución, pero estarán entre 0 y 1.

**Uso práctico.**

`torch.rand` sirve para practicar con datos sintéticos y para simular entradas continuas normalizadas.

---

## 18. Crear tensores aleatorios con `torch.randn`

La función `torch.randn` crea tensores con valores aleatorios distribuidos como una normal estándar.

```python
x = torch.randn(4)

print(x)
print(x.shape)
```

**Interpretación.**

A diferencia de `torch.rand`, aquí pueden aparecer valores positivos y negativos, típicamente alrededor de 0.

**Uso práctico.**

Este tipo de tensor es relevante porque muchas inicializaciones y pruebas en Deep Learning usan valores aleatorios centrados en 0.

---

## 19. Crear enteros aleatorios con `torch.randint`

La función `torch.randint` crea tensores de enteros aleatorios.

```python
x = torch.randint(low=0, high=10, size=(2, 3))

print(x)
print(x.shape)
```

**Interpretación.**

Este código crea una matriz 2 x 3 cuyos valores son enteros aleatorios entre 0 y 9.

El extremo `high=10` no se incluye.

**Uso práctico.**

Puede usarse para crear etiquetas, índices o datos discretos de juguete.

---

## 20. Inspección básica: `shape`

Todo tensor tiene una forma, llamada `shape`.

Ejemplo:

```python
x = torch.tensor([1, 2, 3])

print(x.shape)
```

Resultado conceptual:

```python
torch.Size([3])
```

Otro ejemplo:

```python
y = torch.zeros(2, 3, 4)

print(y.shape)
```

Resultado conceptual:

```python
torch.Size([2, 3, 4])
```

**Interpretación.**

El `shape` indica cuántos elementos hay en cada dimensión.

Aunque el estudio detallado de `shape` se realizará en un bloque posterior, desde la creación de tensores ya es fundamental acostumbrarse a inspeccionarlo.

---

## 21. Inspección básica: `dtype`

Todo tensor tiene un tipo de dato, llamado `dtype`.

Ejemplo:

```python
x = torch.tensor([1, 2, 3])
print(x.dtype)
```

Este tensor probablemente tendrá tipo entero.

En cambio:

```python
y = torch.tensor([1, 2, 3], dtype=torch.float32)
print(y.dtype)
```

Este tensor tendrá tipo flotante de 32 bits.

**Interpretación.**

Los mismos valores numéricos pueden almacenarse con distintos tipos.

En Deep Learning, muchos tensores relevantes se almacenan como `float32`.

---

## 22. Caja de herramientas mínima

Las funciones básicas de creación de tensores estudiadas son:

```python
torch.tensor(...)
torch.zeros(...)
torch.ones(...)
torch.full(...)
torch.arange(...)
torch.linspace(...)
torch.rand(...)
torch.randn(...)
torch.randint(...)
```

Estas funciones son suficientes para construir una primera base práctica de manipulación tensorial.

---

## 23. Errores frecuentes

### 23.1. Confundir listas de Python con tensores de PyTorch

Esto es una lista:

```python
x = [1, 2, 3]
```

Esto es un tensor:

```python
x = torch.tensor([1, 2, 3])
```

La diferencia es importante porque las operaciones de PyTorch se aplican sobre tensores, no sobre listas ordinarias.

---

### 23.2. Crear estructuras irregulares

Esto no define bien una matriz:

```python
x = torch.tensor([
    [1, 2],
    [3]
])
```

El problema es que las filas no tienen la misma longitud.

---

### 23.3. No revisar el `shape`

Muchas confusiones en PyTorch se deben a no mirar la forma del tensor.

Regla práctica:

```python
print(x.shape)
```

debe usarse frecuentemente al practicar.

---

### 23.4. No revisar el `dtype`

Algunas operaciones esperan tensores flotantes, mientras que otras usan enteros.

Regla práctica:

```python
print(x.dtype)
```

permite verificar el tipo de dato.

---

## 24. Micro-galería de creación de tensores

El siguiente bloque resume los principales métodos estudiados.

```python
import torch

print("=== Escalar ===")
a = torch.tensor(5)
print(a)
print("shape:", a.shape)
print("dtype:", a.dtype)

print("\n=== Vector ===")
b = torch.tensor([1, 2, 3])
print(b)
print("shape:", b.shape)
print("dtype:", b.dtype)

print("\n=== Matriz ===")
c = torch.tensor([[1, 2], [3, 4]])
print(c)
print("shape:", c.shape)
print("dtype:", c.dtype)

print("\n=== Tensor 3D ===")
d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(d)
print("shape:", d.shape)
print("dtype:", d.dtype)

print("\n=== Ceros ===")
e = torch.zeros(2, 3)
print(e)
print("shape:", e.shape)
print("dtype:", e.dtype)

print("\n=== Unos ===")
f = torch.ones(2, 3)
print(f)
print("shape:", f.shape)
print("dtype:", f.dtype)

print("\n=== Valor fijo ===")
g = torch.full((2, 3), 7)
print(g)
print("shape:", g.shape)
print("dtype:", g.dtype)

print("\n=== Arange ===")
h = torch.arange(0, 5)
print(h)
print("shape:", h.shape)
print("dtype:", h.dtype)

print("\n=== Linspace ===")
i = torch.linspace(0, 1, steps=5)
print(i)
print("shape:", i.shape)
print("dtype:", i.dtype)

print("\n=== Rand ===")
j = torch.rand(2, 2)
print(j)
print("shape:", j.shape)
print("dtype:", j.dtype)

print("\n=== Randn ===")
k = torch.randn(2, 2)
print(k)
print("shape:", k.shape)
print("dtype:", k.dtype)

print("\n=== Randint ===")
l = torch.randint(low=0, high=10, size=(2, 3))
print(l)
print("shape:", l.shape)
print("dtype:", l.dtype)

print("\n=== Float32 explícito ===")
m = torch.tensor([1, 2, 3], dtype=torch.float32)
print(m)
print("shape:", m.shape)
print("dtype:", m.dtype)
```

Este script puede servir como primera práctica mínima de creación de tensores.

---

## 25. Ejercicios propuestos

### Ejercicio 1

Crear un tensor escalar con el número `12`.

Imprimir:

```python
print(x)
print(x.shape)
print(x.dtype)
```

---

### Ejercicio 2

Crear un vector con los valores:

```python
[4, 7, 9, 10]
```

Indicar su `shape`.

---

### Ejercicio 3

Crear la matriz:

```python
[[1, 2, 3],
 [4, 5, 6]]
```

Indicar su `shape`.

---

### Ejercicio 4

Crear los siguientes tensores:

1. tensor de ceros de forma `(3, 2)`;
2. tensor de unos de forma `(2, 4)`;
3. tensor lleno de `-1` de forma `(2, 2)`.

---

### Ejercicio 5

Usar `torch.arange` para crear:

```python
[0, 1, 2, 3, 4, 5]
```

y luego:

```python
[3, 4, 5, 6, 7]
```

---

### Ejercicio 6

Usar `torch.linspace` para crear 6 puntos entre 0 y 3.

---

### Ejercicio 7

Crear una matriz aleatoria de forma `(3, 3)` usando `torch.rand`.

---

### Ejercicio 8

Crear una matriz de enteros aleatorios de forma `(2, 5)` con valores entre 0 y 4.

---

### Ejercicio 9

Crear dos tensores con los mismos valores `[1, 2, 3]`:

1. uno con tipo entero;
2. otro con tipo `torch.float32`.

Comparar sus `dtype`.

---

### Ejercicio 10

Intentar crear el siguiente tensor irregular:

```python
x = torch.tensor([[1, 2], [3]])
```

Registrar qué ocurre y explicar por qué no corresponde a una estructura tensorial regular.

---

## 26. Interpretación para el proyecto TTM2

La creación de tensores es un paso elemental, pero necesario.

En TTM2 aparecerán tensores que representan:

* features de nodos;
* features de aristas;
* pesos aprendibles;
* kernels;
* canales;
* componentes de tensores esféricos;
* embeddings;
* mensajes en grafos.

Antes de estudiar esas estructuras en detalle, es necesario poder crear, inspeccionar e interpretar tensores simples.

La meta no es quedarse en estos ejemplos, sino usarlos como base para estudiar posteriormente:

* `shape`;
* indexing;
* `reshape` y `view`;
* `stack`;
* `cat`;
* `matmul`;
* módulos de PyTorch;
* DGL;
* message passing.

---

## 27. Síntesis conceptual

La idea central de este apunte es:

> PyTorch representa los objetos numéricos fundamentales del Deep Learning mediante tensores, y crear tensores es el primer paso para poder operar con ellos.

En esta etapa, un tensor debe entenderse como un arreglo multidimensional de números.

Crear tensores implica especificar:

1. qué datos contienen;
2. qué forma tienen;
3. qué tipo de dato almacenan.

---

## 28. Checklist de avance de la sesión

Al cierre de esta parte de la sesión B, se puede registrar el siguiente avance:

* [x] Se introdujo qué es PyTorch.
* [x] Se explicó por qué PyTorch es importante en Deep Learning.
* [x] Se definió operativamente qué es un tensor en PyTorch.
* [x] Se distinguió tensor computacional de tensor geométrico.
* [x] Se estudiaron formas básicas de creación de tensores.
* [x] Se trabajó con tensores desde datos escritos a mano.
* [x] Se estudiaron `zeros`, `ones` y `full`.
* [x] Se estudiaron `arange` y `linspace`.
* [x] Se estudiaron `rand`, `randn` y `randint`.
* [x] Se introdujeron `shape` y `dtype` como propiedades básicas de inspección.
* [x] Se propusieron ejercicios prácticos para consolidar la sesión.

---

## 29. Estado del entregable

Este documento constituye la primera versión formal del apunte técnico:

**PyTorch básico I**

correspondiente a la sesión B de la semana 2.

La versión actual cubre principalmente:

* introducción a PyTorch;
* noción computacional de tensor;
* creación de tensores;
* inspección básica mediante `shape` y `dtype`;
* ejercicios iniciales.

Queda pendiente ampliar este mismo apunte, o crear subsecciones posteriores, para cubrir en detalle:

* `shape`;
* indexing;
* `reshape/view`;
* `stack`;
* `cat`;
* `matmul`.

---

## 30. Cierre

Este apunte documenta el primer avance técnico concreto del Carril B en la semana 2. Su valor dentro del proyecto es preparar la base mínima para que el estudio de TTM2 no quede bloqueado por dificultades elementales de PyTorch.

La creación de tensores es una habilidad básica, pero estructural: permite empezar a construir micro-pruebas, verificar operaciones y desarrollar una lectura más fluida del código que implementa arquitecturas de Deep Learning.