# Tutorial: A practical introduction to Tensorflow on Sagemaker

This repo contains Juypyter notebooks that guide you through deep learning development process on AWS Sagemaker. During this tutorial, you will learn how to exploit services offered by Sagemaker such as [pipe input mode](https://aws.amazon.com/blogs/machine-learning/using-pipe-input-mode-for-amazon-sagemaker-algorithms/), [automatic model tuning](https://aws.amazon.com/blogs/aws/sagemaker-automatic-model-tuning/) and [sagemaker.tensorflow](https://docs.aws.amazon.com/sagemaker/latest/dg/tf-example1-train.html). 

A tutorial accompanies these Jupiter notebooks. You can find this tutorial [here]().

A brief explenation of the contents of this repo: 

* `cnn_fashion_mnist.py`: entry_point file which which includes the definition of the Tensorflow model + describes how data is fed into the model.
* `create_fashion_tfrecords.ipynb`: prepare the dataset for streaming by converting the dataset into TFRecords.
* `tune_fashion_network.ipynb`: define and tune the tensorflow model using the automatic model tuning service from Sagemaker.
* `train_deploy_fashion_network.ipynb`: train and deploy the final model design using Sagemaker.

