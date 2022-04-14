____INSTALLATION:____

To install the required packages and GPU support please load the Anaconda environment  "execute_models" in this folder:

	1) conda env create -f environment_extract.yml
	2) conda activate execute_models
	3) conda env list 

____EXECUTE CODE____

- To exercise target models and MagNet and Squeezer detectors on normal and andversarial image sets:
	
	- main_mnist_1.py (target model: Carlini)
	- main_mnist_2.py (target model: Cleverhans)
	- main_cifar.py   (target model: DenseNet)
		
	These scripts can be used to extract the deep feature of the model during the classification of adversarial and normal image (see the comments in the code for 	more details). The scripts reproduce the methodology steps detailed in []

- To execute examples of attacks generations scripts used to generate adversarial image sets:  
	
	- generate_bim_mnist_1.py      (target model: Carlini)
	- generate_carlini_mnist_2.py  (target model: Cleverhans)
	- genarate_fgsm_cifar.py       (target model: DenseNet)     	 

____TARGET MODELS____

## Dataset: MNIST

| Model Name | # Trainable Parameters  | Testing Accuracy |
|------------|-------------------------|------------------|
| Cleverhans |  710,218                |     0.9919       |
| Carlini    |  312,202                |     0.9943       |

## Dataset: CIFAR-10

|      Model Name     |  # Trainable Parameters  | Testing Accuracy | 
|---------------------|--------------------------|------------------|
| DenseNet(L=40,k=12) | 1,019,722                |     0.9484       |  



