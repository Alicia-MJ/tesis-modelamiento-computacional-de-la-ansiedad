# Tesis para obtener el grado de Lic. Psicología de la UNAM
## El estudio computacional de la ansiedad con algoritmos de inteligencia artificial

Neuro-Nav (Juliani, A. et. al, 2022), una librería de código abierto para aprendizaje por refuerzo (RL) neronalmente plausible, fue usada para programar la tarea experimental y como base para programar y desarrollar los modelos de RL que se usaron en el proyecto de tesis.   

Las principales modificaciones añadidas a la librería son los siguientes:  

- Al modelo TDSR se le incluyó la variación B-pessimisitc en la función de actualización de las representaciones sucesoras.
- Se creó el modelo TDSR_RP para incluir en el modelo SR diferentes tasas de aprendizajes para los castigos y las recompensas. 
- DynaSR - refieriéndose a la implementación del modelo Dyna B-pessimistic SR, usando como base el modelo TDSR. 
- DynaSR_RP - refieriéndose a la implementación del modelo Dyna \alpha-SR model implementation, usando como base el modelo TDSR_RP.

Ver los [agentes](./agents) para más información.

Además, se eliminaron los archivos de código y agentes que no se usaron para el proyecto de tesis. 

## Poster presentado en MAIN 2024

"The Impact of Punishment Sensitivity and Learning Rate on Anxiety: A Computational Modeling Approach in a Sequential Evaluation Task".

Ver abstract [here](https://www.main2024.org/abstracts).

## Notebook con los experimentos

Un Notebook de Google Colab Notebook las simulaciones hechas en la tarea experimental para el proyecto de tesis. 

Ver [Notebook.ipynb](./Notebook.ipynb) para más información.


## Requerimientos

Requirimientos para poder correr el notebook [aquí](./setup.py).





## Referencia


* Juliani, A., Barnett, S., Davis, B., Sereno, M., & Momennejad, I. (2022). Neuro-Nav: A Library for Neurally-Plausible Reinforcement Learning. The 5th Multidisciplinary Conference on Reinforcement Learning and Decision Making.


