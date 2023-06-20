# Technical notes 
- The order in which vertices of a rectangle are given is: left bottom, right bottom, left top, right top. 

# Room mechanics
Room is a 2d np array with values from 0 to 1 representing dirt and value 2 representing a wall. 

## Broom movement

During the movement of a broom from *p1* to *p2* a rectangle of width predefined in config file is created.

![image](https://github.com/gournge/optymalizacja-sprzatania/assets/81859727/6c2284b5-e8c3-4148-b244-aa1f772ef76a)

