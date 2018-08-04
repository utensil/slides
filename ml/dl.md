# Deep Learning

---

### Going Deep

note:

- http://slides.com/beamandrew/deep-learning-101#/
- http://donsoft.io/intro-to-deeplearning
- https://github.com/kristjankorjus/applied-deep-learning-resources
- https://github.com/m2dsupsdlclass/lectures-labs
- https://github.com/InfolabAI/DeepLearning
- https://github.com/roboticcam/machine-learning-notes
- https://www.slideshare.net/JrgenSandig/neural-networks-and-deep-learning

***

<!-- .slide: data-background-iframe="http://slides.com/beamandrew/deep-learning-101-10#/3" data-background-interactive -->

***

<!-- .slide: data-background-iframe="http://slides.com/beamandrew/deep-learning-101#/35" data-background-interactive -->

***

<!-- .slide: data-background-iframe="http://cs231n.github.io/assets/conv-demo/index.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/01_intro_to_deep_learning/" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/02_backprop/" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/03_recommender_systems" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/04_conv_nets" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/05_conv_nets_2" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/06_deep_nlp" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/07_deep_nlp_2" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/08_expressivity_optimization_generalization" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/09_imbalanced_classif_metric_learning" data-background-interactive -->

<!-- .slide: data-background-iframe="https://m2dsupsdlclass.github.io/lectures-labs/slides/10_unsupervised_generative_models" data-background-interactive -->

---

### Different Ways to understand Deep Learning

***

<!-- .slide: style="font-size:32px;" -->

#### Three major narratives to understand Deep Learning

- **Neuroscience Narrative**: drawing analogies to biology
- **Probabilistic Narrative**: interprets neural networks as finding latent variables
- **Representations Narrative**: centered on transformations of data and the manifold hypothesis
  - deep learning studies a connection between optimization and functional programming
    - the representations narrative in deep learning corresponds to type theory in functional programming

[Neural Networks, Types, and Functional Programming](http://colah.github.io/posts/2015-09-NN-Types-FP/) <!-- .element: class="figcaption" -->

***

<!-- .slide: style="font-size:32px;" -->

#### More ways to understand Deep Learning


- [New Theory Cracks Open the Black Box of Deep Learning](https://www.wired.com/story/new-theory-deep-learning/)
  - [Chinese Version](https://zhuanlan.zhihu.com/p/29579424)
- [The Holographic Principle: Why Deep Learning Works](https://medium.com/intuitionmachine/the-holographic-principle-and-deep-learning-52c2d6da8d9)
  - [Chinese Version](https://cloud.tencent.com/developer/article/1164216)
- [Deep Learning and Quantum Entanglement: Fundamental Connections with Implications to Network Design](https://arxiv.org/abs/1704.01552)
  - [Chinese Version](http://www.sohu.com/a/133017705_465975)

Note:

Source: https://www.zhihu.com/question/54902742/answer/300037431

***

<!-- .slide: data-background-video="http://donsoft.io/intro-to-deeplearning/videos/neuron.mp4" data-background-interactive -->

***

<!-- .slide: data-background-video="http://donsoft.io/intro-to-deeplearning/videos/neural_networks.mp4" data-background-interactive -->

***

#### Neuroscience Narrative

![1940's 1950's - Dedicated vs. universal - Analog vs. digital - Decimal vs. binary - Wired vs. memory-based programming - S...](https://image.slidesharecdn.com/20141030ibmforretaping-150210111659-conversion-gate01/95/what-the-brain-says-about-machine-intelligence-2-638.jpg?cb=1424713709) 

[What the Brain says about Machine Intelligence](https://www.slideshare.net/numenta/what-the-brain-says-about-machine-intelligence) <!-- .element: class="figcaption see-also" -->

***

#### Neuroscience Narrative (Cont.)

![2010's 2020's The Birth of Machine Intelligence - Specific vs. universal algorithms - Mathematical vs. memory-based - Batc...](https://image.slidesharecdn.com/20141030ibmforretaping-150210111659-conversion-gate01/95/what-the-brain-says-about-machine-intelligence-3-638.jpg?cb=1424713709) 

[What the Brain says about Machine Intelligence](https://www.slideshare.net/numenta/what-the-brain-says-about-machine-intelligence) <!-- .element: class="figcaption see-also" -->

***

<!-- .slide: data-background-iframe="//www.slideshare.net/slideshow/embed_code/key/A6pXGOUXNgDYrx" data-background-interactive -->

note:

https://www.slideshare.net/numenta/2014-10-17-numenta-workshop

***

<!-- .slide: style="font-size:32px;" -->

#### Probabilistic Narrative

- [Geometric Understanding of Deep Learning](https://arxiv.org/abs/1805.10451)
  - [I:  Manifold distribution law](https://mp.weixin.qq.com/s/onqVbGBdS5pfM3itdBP8eQ)
  - [II:Upper limit of learning ability](https://mp.weixin.qq.com/s/zqXqDI6aV8gv6KOUPQB84Q)
  - [III: Geometric View of probability transformation](https://mp.weixin.qq.com/s/z1caI6P34xUI7UC9mHjI-A)
- [A Geometric View of Optimal Transportation and Generative Model](https://arxiv.org/abs/1710.05488)
  - [GAN with the wind](https://mp.weixin.qq.com/s/7O0AKIUVYK7HRyvdRbUVkg)
  - [See through the black box of W-GAN I](https://mp.weixin.qq.com/s/trvMOTXNs7L6fSmTkZXwsA)
  - [See through the black box of W-GAN II](https://mp.weixin.qq.com/s/thcxsBVttSIEzVNLQlAVCA)
  - [See through the black box of W-GAN III](https://mp.weixin.qq.com/s/Jx0o17CwlIVcRV22PXk4wQ)
  - [Related Zhihu Question](https://www.zhihu.com/question/67080147)

---

### Deep Learning - The Straight Dope

http://gluon.mxnet.io/

***

#### Simple Networks

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter02_supervised-learning/linear-regression-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter02_supervised-learning/perceptron.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter03_deep-neural-networks/mlp-scratch.html" data-background-interactive -->

***

#### More than Networks

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter02_supervised-learning/regularization-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter02_supervised-learning/environment.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter03_deep-neural-networks/mlp-dropout-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter18_variational-methods-and-uncertainty/bayes-by-backprop.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/optimization-intro.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/gd-sgd-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/momentum-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/adagrad-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/rmsprop-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/rmsprop-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter06_optimization/adam-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter08_computer-vision/fine-tuning.html" data-background-interactive -->

***

#### Deep Learning Networks

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter04_convolutional-neural-networks/cnn-batch-norm-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter05_recurrent-neural-networks/simple-rnn.html" data-background-interactive -->
***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter05_recurrent-neural-networks/lstm-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter05_recurrent-neural-networks/gru-scratch.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter14_generative-adversarial-networks/gan-intro.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter14_generative-adversarial-networks/dcgan.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter14_generative-adversarial-networks/pixel2pixel.html" data-background-interactive -->

***

#### Applications

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter08_computer-vision/object-detection.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//gluon.mxnet.io/chapter08_computer-vision/visual-question-answer.html" data-background-interactive -->

---

### Libs

***

<!-- .slide: style="font-size: 28px" -->

#### Tensorflow

- [Data science Python notebooks: TensorFlow Tutorials](https://github.com/donnemartin/data-science-ipython-notebooks#tensor-flow-tutorials)
- [Tensorflow Tutorials using Jupyter Notebook](https://github.com/sjchoi86/Tensorflow-101)
- [Effective TensorFlow](https://github.com/vahidk/EffectiveTensorflow)
- [TensorFlow-Tutorials: From LR to GAN](https://github.com/nlintz/TensorFlow-Tutorials)
- [TensorFlow in Chinese](https://github.com/lawlite19/MachineLearning_TensorFlow)
- [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)

***

<!-- .slide: style="font-size: 28px" -->

#### Other than Tensorflow

- [PyTorch](https://github.com/pytorch/pytorch)
- [MXNet](https://github.com/apache/incubator-mxnet)
- [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
- [DLib](http://dlib.net/ml.html)
- [Caffe2](https://github.com/caffe2/caffe2)
- [Deep Learning Framework Examples](https://github.com/ilkarman/DeepLearningFrameworks)

***

<!-- .slide: style="font-size: 28px" -->

#### Toolkits

- [mxnet-finetuner](https://github.com/knjcode/mxnet-finetuner)
- [ImageAI](https://github.com/OlafenwaMoses/ImageAI)
- [GluonCV — Deep Learning Toolkit for Computer Vision](https://medium.com/apache-mxnet/gluoncv-deep-learning-toolkit-for-computer-vision-9218a907e8da)
- [GluonNLP — Deep Learning Toolkit for Natural Language Processing](https://medium.com/apache-mxnet/gluonnlp-deep-learning-toolkit-for-natural-language-processing-98e684131c8a)

---

### Resources

***

<!-- .slide: style="font-size: 28px" -->

#### Datasets

- All
  - https://github.com/caesar0301/awesome-public-datasets
  - http://academictorrents.com/
- CV
  - http://deeplearning.net/datasets
- NLP
  - http://universaldependencies.org/
  - https://github.com/Breakend/DialogDatasets
- Go
  - https://github.com/yenw/computer-go-dataset

***

<!-- .slide: style="font-size: 28px" -->

#### Notebooks

- [TensorFlow Exercises: focusing on the comparison with NumPy](https://github.com/Kyubyong/tensorflow-exercises)
- [TensorFlow Tutorial - used by Nvidia](https://github.com/alrojo/tensorflow-tutorial)
- [Python Machine Learning Notebooks](https://github.com/tirthajyoti/PythonMachineLearning)
- [Data science Python notebooks](https://github.com/donnemartin/data-science-ipython-notebooks)
- [Python Data Science Handbook: full text in Jupyter Notebooks](https://github.com/jakevdp/PythonDataScienceHandbook)
- [Machine Learning & Deep Learning Tutorials](https://github.com/ujjwalkarn/Machine-Learning-Tutorials)

***

<!-- .slide: style="font-size: 28px" -->

#### Implementations

- [Minimal Machine learning algorithms](https://github.com/rushter/MLAlgorithms/)
- [Machine Learning From Scratc](https://github.com/eriklindernoren/ML-From-Scratch)
- [tiny-dnn](https://github.com/tiny-dnn/tiny-dnn) & [MiniDNN](https://github.com/yixuan/MiniDNN)

***

<!-- .slide: style="font-size: 28px" -->

#### Docker Images

- https://github.com/waleedka/modern-deep-learning-docker
- https://github.com/ufoym/deepo
- https://github.com/bethgelab/docker
- https://github.com/utensil/dockerfiles
- https://www.docker-cn.com/registry-mirror
- https://www.daocloud.io/mirror


---

#### Reinforcement Learning

***

<!-- .slide: style="font-size: 28px" -->

#### Implementations

- [Simple Beginner’s guide to Reinforcement Learning & its implementation](https://www.analyticsvidhya.com/blog/2017/01/introduction-to-reinforcement-learning-implementation/)
- [Deep Deterministic Policy Gradient (DDPG) (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-2-DDPG/)
- [Reinforcement learning tutorial using Python and Keras](http://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/)
- [TensorForce](https://github.com/reinforceio/tensorforce)
- [REINFORCEjs](https://github.com/karpathy/reinforcej)

***

<!-- .slide: style="font-size: 28px" -->

#### Environments

- [OpenAI Gym](https://github.com/openai/gym)
  - CartPole-v0
- [Retro Games in Gym](https://github.com/openai/retro)
  - Airstriker
- [PyGame Learning Environment (PLE)](http://pygame-learning-environment.readthedocs.io/en/latest/index.html)
  - [Keras-FlappyBird](https://github.com/yanpanlau/Keras-FlappyBird)
- [osim-rl](http://osim-rl.stanford.edu/docs/quickstart/)
  - L2RunEnv

---

### Visualization

***

<!-- .slide: style="font-size: 28px" -->

- http://playground.tensorflow.org
- https://deeplearnjs.org/demos/model-builder/
- http://cs.stanford.edu/people/karpathy/convnetjs/
- http://cs231n.github.io/neural-networks-3/
- http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- http://colah.github.io/
- https://colah.github.io/posts/2014-07-Understanding-Convolutions/
- https://github.com/keplr-io/hera
- https://github.com/awslabs/mxboard
- https://github.com/slundberg/shap

***

<!-- .slide: style="font-size: 28px" -->

- http://colah.github.io/posts/2015-09-NN-Types-FP/
- http://colah.github.io/posts/2015-01-Visualizing-Representations/
- http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
- http://colah.github.io/posts/2014-10-Visualizing-MNIST/
- https://zhuanlan.zhihu.com/p/27204291
- https://blog.jakuba.net/2017/05/30/tensorflow-visualization.html

***

<!-- .slide: style="font-size: 28px" -->

- https://github.com/ethereon/netscope
- https://www.zhihu.com/question/26006703/answer/135825424
- http://josephpcohen.com/w/visualizing-cnn-architectures-side-by-side-with-mxnet/
- https://github.com/apache/incubator-mxnet/tree/master/example/image-classification

***

<!-- .slide: style="font-size: 28px" -->

- http://nbviewer.jupyter.org/github/stephencwelch/Neural-Networks-Demystified/tree/master/
- https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
- http://www.scipy-lectures.org/intro/numpy/numpy.html#indexing-and-slicing
- http://scikit-learn.org/stable/modules/tree.html
- https://distill.pub/2017/momentum/
- https://github.com/distillpub/post--feature-visualization
- https://distill.pub/2017/feature-visualization/
- https://bondifrench.github.io/ml-in-js/
- http://nbviewer.jupyter.org/github/lightning-viz/lightning-example-notebooks/blob/master/index.ipynb

---

### TensorFlow Exercises

<small>focusing on the comparison with NumPy</small>

https://github.com/Kyubyong/tensorflow-exercises

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Constants_Sequences_and_Random_Values_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Graph_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Variables_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Reading_Data_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Tensor_Transformations_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Math_Part1_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Math_Part2_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Math_Part3_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Control_Flow_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Sparse_Tensors-Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Neural_Network_Part1_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Neural_Network_Part2_Solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Seq2Seq_solutions.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="https://nbviewer.jupyter.org/github/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb" data-background-interactive -->

---

### TensorFlow Tutorial

<small>used by Nvidia</small>

https://github.com/alrojo/tensorflow-tutorial

***

<!-- .slide: data-background-iframe="//nbviewer.jupyter.org/github/alrojo/tensorflow-tutorial/blob/master/lab1_FFN/lab1_FFN.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//nbviewer.jupyter.org/github/alrojo/tensorflow-tutorial/blob/master/lab2_CNN/lab2_CNN.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//nbviewer.jupyter.org/github/alrojo/tensorflow-tutorial/blob/master/lab3_RNN/lab3_RNN.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//nbviewer.jupyter.org/github/alrojo/tensorflow-tutorial/blob/master/lab4_Kaggle/lab4_Kaggle.ipynb" data-background-interactive -->

***

<!-- .slide: data-background-iframe="//nbviewer.jupyter.org/github/alrojo/tensorflow-tutorial/blob/master/lab5_AE/lab5_AE.ipynb" data-background-interactive -->

---

### Understanding LSTM Networks

- http://colah.github.io/posts/2015-08-Understanding-LSTMs
- https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714
- [LSTM and GRU -- Formula Summary](https://isaacchanghau.github.io/post/lstm-gru-formula/)

***

<!-- #####.slide: data-background-iframe="//colah.github.io/posts/2015-08-Understanding-LSTMs/" data-background-interactive -->
