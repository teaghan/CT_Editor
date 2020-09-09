# CT Editor
Using adversarial learning to insert realistic looking tumours into pre-existing patient CT scans

## Dependencies

-[PyTorch](http://pytorch.org/): `pip install torch torchvision`

-h5py: `pip install h5py`

-scipy: `pip install scipy`

## Adversarial Learning for CT Editing

This project aims to implement a method similar to [Improving the Realism of Synthetic Images](https://machinelearning.apple.com/2017/07/07/GAN.html) in order to achieve the ability to take exisiting patient CT images and insert tumours into hand-chosen locations.
                                   
## Getting Started ##

1. Download the training, validation, and test data from __ and place them into the [data directory](./data).
    
2. The model architecture and hyper-parameters are set within configuration file in [the config directory](./configs). For instance, I have already created the [base line configuration file](./configs/hlt_to_nod_1.ini). If creating a new model, I suggest copying this one and changing the parameters
  
3. Using this model as my example, from the main CT_Editor directory, you can run `python train_network.py hlt_to_nod_1 -v 2000 -ct 15` which will train your model displaying the progress every 2000 batch iterations and saves the model every 15 minutes. This same command will continue training the network if you already have the model saved in the [model directory](./models) from previous training. (Note that the training takes approximately 27 hours on GPU). Alternatively, if operating on compute-canada see [this script](./scripts/hlt_to_nod_1.sh) for the training. It allows faster data loading throughout training.
  
4. The [Analysis notebook](./evaluate_samples.ipynb) allows for the evaluation of the network throughout the training process

5. The developed [GUI](./qt/) can be used to easily interact with the editor network.



## Citing this work

Checkout our [Technical Note](https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14437). When using this tool, please consider citing our work. For example, here's the BibTeX:

```
@article{OBriain2020,
  doi = {10.1002/mp.14437},
  url = {https://doi.org/10.1002/mp.14437},
  year = {2020},
  month = aug,
  publisher = {Wiley},
  author = {Teaghan B. O{\textquotesingle}Briain and Kwang Moo Yi and Magdalena Bazalova-Carter},
  title = {Technical Note: Synthesizing of lung tumors in computed tomography images},
  journal = {Medical Physics}
}
```
