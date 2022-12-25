Particle Replay Buffer For Energy Based Model
=======
# Introduction 
Although the EBM model does not have any constraint, the EBM model does not work well on the image dataset, and the best FID on the CIFAR10 dataset is 30. One of the main reasons is that the Langevien sampling is not well harnessed in high dimensions, so this project proposes a replay buffer that can further update the buffer with the update of the model. This initiative can, first of all, reduce the number of steps of Langevein sampling, improve the training speed, and, most importantly, make the distribution of the buffer more closely match the distribution corresponding to the energy equation, which can, to a certain extent, weaken the possible errors brought by Langevein sampling.

```shell
# Install all the dependencies for the frontend
pip install -r requirements.txt
```

The basic usage for training is
```sh
python train_ebm.py
```
