# Deep Leakage From Gradients
This repository includes an unofficial implementation of the paper entitled: "Deep Leakage from Gradients"
# How to use the code
To run the DLG algorithm on one of the 50,000 images in the CIFAR100 dataset use:

      python3 main.py --index <image index ranging from 0 to 49,999>
      
To visualize the gradient matching loss when adding gaussian noise to the gradients use:

      python3 noise.py -- index <image index ranging from 0 to 49,999  

All the dependencies needed to run the code can be found at  *requirements.txt*
      
