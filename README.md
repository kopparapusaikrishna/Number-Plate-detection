# Car Number-Plate-detection

Number plate detection allows computers to read and interpret vehicle number plates from digital images. It involves the use of image processing techniques to detect, segment, and recognize characters on license plates, which can be used for various applications such as traffic monitoring, law enforcement, and toll collection.

The number plate detection problem is complex and challenging, as it can appear in different sizes, shapes, orientations, and lighting conditions, and can be partially obscured or blurred. In addition, number plates can contain different fonts, colors, and background designs, which further complicates the task of detecting and recognizing them accurately and reliably.

In this project we first used CNN for identifying number plates, but it wasn’t performing well. So, we then shifted to the Yolov5 model to identify the number plates. The dataset that we used contains the pictures of car number plates taken from different angles and different lighting systems.

We design an application where one can upload the image containing a car’s number plate to see the output our trained model gives.

The architecture of our project demands two layers.
Front end
Back end
The front end of the project is handled by “html”, javascript and the backend is swiftly handled by “Python”, using “Flask” for interaction between ends.
