This repository was made to document a group project under the supervision of Silesian Politechnic. 

The deeper application of this project is to see if during a process of collective displacement of objects it is optimal to centralise piles or rather diversify it into making smaller heaps in seperate sub-rooms.  

The model atttempts to find an optimal sequence of broom movements to clean a room with a given shape. The sequence is found by sequentially applying the NEAT algorithm (```NeatCleaningModel.py```), which has been trained on random rooms (```RoomGeneration.py```) first with naively chosen mounds of dirt (by applying clustering algorithms - intuitvely we should sweep the broom to where dirt is mostly clustered.) After the NEAT algorithm has been developed, a module (```PlanMoundsModel.py```) for planning mounds (mounds in the room mechanics act like holes in the floor consuming the dirt) is trained on random rooms and based on evaluations of the NEAT Cleaning Model. 


# Room mechanics
Room is a 2d np array with values from 0 to 1 representing dirt and value 2 representing a wall. 

## Broom movement

During the movement of a broom from *p1* to *p2* a rectangle of width predefined in config file is created.

![image](https://github.com/gournge/optymalizacja-sprzatania/assets/81859727/6c2284b5-e8c3-4148-b244-aa1f772ef76a)

It is a reference for how dirt is transformed. During the movement we establish a critical point where broom's capacity is exceeded. Based on its distance to the beginning of movement we create two rectangles on the side (only if broom capacity is actually exceeded) and one in front. 

If the capacity is not exceeded we move all the dirt to the front rectangle - if the capacity of the front one is exceeded, we distribute the dirt back to the original rectangle (yellow in the picture below.) 
If the capacity of the broom is exceeded we create two side rectangles and split 

![clean optim figure 2](https://github.com/gournge/cleaning-optimization/assets/81859727/dc02cb13-4e7a-4e18-9ef2-92097dc7b6fc)

Moving dirt to a rectangle means randomly distributing it in 
