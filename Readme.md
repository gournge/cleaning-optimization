# Project description 

How do you plan the logistics of moving items around in a large usable space? Initially, it might seem appropriate to partition the space into smaller sections in which movement would be implicitly manageable. In our strategy, however, we make quite considerable use of the assumption that there is no easier method of alternative space allocation or movement organisation. The main objective of this work is to validate this assumption, i.e. to investigate whether there is a better method than the default approach in general. 

The deeper application of this project is to see if during a process of collective displacement of objects (randomly arranged in the beginning) it is optimal to centralise piles or rather diversify it into making smaller heaps in seperate sub-rooms. Another theoretical application is to compare the results returned by the algorithm - which describe how to plan moving dirt around - to what existing clustering algorithms return. 

The model atttempts to find an optimal sequence of broom movements to clean a room with a given shape. The sequence is found by sequentially applying the NEAT algorithm (```NeatCleaningModel.py```), which has been trained on random rooms (```RoomGeneration.py```) first with naively chosen mounds of dirt (by applying clustering algorithms - intuitvely we should sweep the broom to where dirt is mostly clustered.) After the NEAT algorithm has been developed, a module (```PlanMoundsModel.py```) for planning mounds (mounds in the room mechanics act like holes in the floor consuming the dirt) is trained on random rooms and based on evaluations of the NEAT Cleaning Model. 


# Room mechanics
Room is a 2d np array with values from 0 to 1 representing dirt and value 2 representing a wall. 

## Broom movement

During the movement of a broom from *p1* to *p2* a rectangle of width predefined in config file is created.

![image](https://github.com/gournge/optymalizacja-sprzatania/assets/81859727/6c2284b5-e8c3-4148-b244-aa1f772ef76a)
