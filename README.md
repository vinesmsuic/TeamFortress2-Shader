# TeamFortress2-Shader
A Rasterization+Raytracing renderer toy. 

All core functions were implemented with purely C++ (no OpenGL).

This is my UWaterloo CS488/688 project (Free topic). The project features different shaders in raytracing mode:
* [Half Lambert](https://developer.valvesoftware.com/wiki/Half_Lambert)
* [Phong Shading](https://users.cs.northwestern.edu/~ago820/cs395/Papers/Phong_1975.pdf)
* Fresnel Reflectance
* Toon Shading
* [X-toon: An extended toon shader](https://dl.acm.org/doi/10.1145/1124728.1124749)
* [Illustrative Rendering in Team Fortress 2](https://dl.acm.org/doi/10.1145/1274871.1274883)

## Setup
```shell
./get_media.sh
```
It should produce the media resources for the project.

Now you can simply build the project.

## How to run the project

### Objective 1

Objective 1: Implement a Naive toon shading / cel shading using intensity.

Description: The shading function is implemented in `shadeToonSimple` method.

![img](https://i.imgur.com/yNGJLSY.png)

To run the project for objective 1:

```shell
CS488 ../media/blueball.obj
```

or

```shell
CS488 ../media/model_sarah.obj
```

### Objective 2

Objective 2: Implement Half Lambert to prevent the rear of an object with Lambertian looking too flat.

Description: The shading function is implemented in `shadeHalfLambertShadow` method.

![img](https://i.imgur.com/Lnc3Tn0.png)

To run the project for objective 2:

```shell
CS488 ../media/cornellbox-half.obj
```

### Objective 3 

Objective 3: Implement Phong shading for glossy surfaces.

Description: The shading function is implemented in `shadePhong` method.

![img](https://i.imgur.com/x7iQ1mx.png)

To run the project for objective 3:

```shell
CS488 ../media/custom-spheres.obj
```

### Objective 4

Objective 4: Implement Fresnel Reflectance so both reflection and refraction occur at interface.

![img](https://i.imgur.com/Oqa3Oor.png)

To run the project for objective 4:

```shell
CS488 ../media/custom-fresnel5.obj
```

### Objective 5

Objective 5: Implement X-toon the extended toon shader to create silhouette feelings for the rendered objects.

![img](https://i.imgur.com/hGcLmxy.png)

To run the project for objective 5:

```shell
CS488 ../media/bunny-xtoon.obj
```

or

```sh
CS488 ../media/model_sarah_anti-bully.obj
```





### Objective 6

Objective 6: Implement the View Independent lighting of Team Fortress 2 shading.

![img](https://i.imgur.com/zwioJiP.png)

To run the project for objective 6:

```shell
CS488 ../media/heavyweaponsmanred1.obj
```

### Objective 7

Objective 7: Implement the View Dependent lighting of Team Fortress 2 shading.

![img](https://i.imgur.com/jJUNFxG.png)

To run the project for objective 7:

```shell
CS488 ../media/heavyweaponsmanred2.obj
```

### Objective 8

Objective 8: Combining the major components of Team Fortress 2 shading and evaluating the effect of parameters

![img](https://i.imgur.com/NE3BXsR.png)

To run the project for objective 8:

```shell
CS488 ../media/heavyweaponsmanred3.obj
```



I made another version which loads the normal map of the model but I am not sure if I implemented it correctly.

```shell
CS488 ../media/heavyweaponsmanred4.obj
```



## Implementation Detail

For math equations, please refer to my report:

* [Reimplementing Shading from the Video Game “Team Fortress 2”](https://drive.google.com/file/d/1shZdfo1f4UPAJZjE8OQlPA4EvR2oWjXy/view?usp=sharing)



## Credit 

All the TF2-related models / textures are downloaded from the internet. Links include:

* [The Heavy Weapons Guy - Download Free 3D model by Rage_Models (@ragemodels) be2a866\] (sketchfab.com)](https://sketchfab.com/3d-models/the-heavy-weapons-guy-be2a866406584f418168f1aca4309aee)
* [PC / Computer - Team Fortress 2 - Heavy - The Models Resource (models-resource.com)](https://www.models-resource.com/pc_computer/teamfortress2/model/6571/)
* [PC / Computer - Team Fortress 2 - Minigun - The Models Resource (models-resource.com)](https://www.models-resource.com/pc_computer/teamfortress2/model/35825/)
* [Team Fortress 2 GLSL shader - Game Engine / Game Engine Resources - Blender Artists Community](https://blenderartists.org/t/team-fortress-2-glsl-shader/580614/5)

other models are either drawn or constructed by me using Blender 2.83.



## Video Demo

<iframe width="560" height="315" src="https://www.youtube.com/embed/IasiltGi1b0?si=KdCls18T4uzpOAE5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>