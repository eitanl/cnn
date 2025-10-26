# cnn
a simple cnn, implemented in python using PyTorch


* תמונת קלט 28x28 מונוכרומטית
  * שכבת קונב' ראשונה: **288** + 32 פרמטרים (32 קונב' 3x3) 
* טנזור 32x14x14
  * שכבת קונב' שניה: **18,432** + 64 פרמטרים (64 קונב' 3x3x32)
* טנזור 64x7x7
* משטיחים לווקטור באורך 3,136
  * שכבה FC / לינארית ראשונה: **401,408** + 128 פרמטרים
* ווקטור באורך 128
  * שכבה FC / לינארית שניה: **1,280** + 10 פרמטרים
* פלט באורך 10 (אחד לכל סיפרה אפשרית)

Training and testing is done on the MNIST dataset.

### How to use
* create a virtual environment (recommended) and install the requirements - using the provided scripts
* run cnn.py - it will do
  * model set up
  * training (from scratch )
  * testing / evaluation
