# A Machine Learning Tutorial

---

### The Origin

- [Mathematica](https://www.wolfram.com/mathematica/)[*](https://github.com/utensil/mathematica-notebooks) (2000) 
- [Neural Networks](https://book.douban.com/subject/1115600/) (2002)
- [NuPIC](https://github.com/numenta/nupic) and [NuPIC Core](https://github.com/numenta/nupic.core/pulls?utf8=%E2%9C%93&q=author%3Autensil) (2014)
- [kaggle: Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson) (2014)
- Dive in (2016)

Note:

- [Robot Movies](https://www.ranker.com/list/best-artificial-intelligence-movies-list/all-genre-movies-lists)
  - http://www.artofthetitle.com/title/eva/
- [Amazing Ants](https://book.douban.com/subject/1044309/) (2002)

Dive in inclues Julia, Geometric Algebra, Keras and Risk Control.

---

### Overview

***

#### Three Types of Machine Learning

![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_01.png?raw=true) [Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

Note:

| Supervised Learning | Unsupervised Learning | Reinforcement Learning |
|---------------------|-----------------------|------------------------|
| ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_02.png?raw=true) | ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_06.png?raw=true) | ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_05.png?raw=true)

[Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

***

#### Supervised Learning

![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_02.png?raw=true) [Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

***

#### Supervised Learning (Cont.)

| Classification | Regression |
|---------------------|-----------------------|
| ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_03.png?raw=true)  <!-- .element: class="img-300" -->  | ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_04.png?raw=true)  <!-- .element: class="img-300" --> |

[Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

***

#### Unsupervised Learning

| Clustering | Dimensionality Reduction |
|---------------------|-----------------------|
| ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_06.png?raw=true)  <!-- .element: class="img-300" -->  | ![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_07.png?raw=true)  <!-- .element: class="img-300" --> |

[Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

Note:

http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html

***

#### Reinforcement Learning

![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_05.png?raw=true)

[Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

***

<!-- .slide: data-background-iframe="https://www.youtube.com/embed/gn4nRCC9TwQ" data-background-interactive -->

***

#### A roadmap for building machine learning systems


![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch01/images/01_09.png?raw=true) [Python Machine Learning: Chapter 1 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch01/ch01.ipynb)  <!-- .element: class="figcaption" -->

---

### Basic Concepts in Data Science

***

#### What is Data Science?

![](https://i2.wp.com/blog.operasolutions.com/Portals/213893/images/OS_SignalCentral%20What%20Is%20Data%20Science_GP_2014_V1.2_Venn%20Diagram.jpg?zoom=2)   <!-- .element: class="img-500" --> [DATA SCIENCE & SCIENTISTS](https://science2knowledge.wordpress.com/data-science-scientists/)   <!-- .element: class="figcaption" -->

***

#### Epicycles of Analysis <!-- .element: style="font-size: 42px" -->

![](https://i0.wp.com/blog.operasolutions.com/Portals/213893/images/OS_SignalCentral%20What%20Is%20Data%20Science_GP_2014_V1.2_The%20Data%20Science%20Process.jpg?zoom=2)   <!-- .element: class="img-550" -->  [DATA SCIENCE & SCIENTISTS](https://science2knowledge.wordpress.com/data-science-scientists/)   <!-- .element: class="figcaption" -->

Note:

- Stating and refining the question
- Exploring the data
- Building formal statistical models
- Interpreting the results
- Communicating the results

https://gohighbrow.com/the-data-analysis-epicycle/
https://bookdown.org/rdpeng/artofdatascience/epicycle-of-analysis.html

![](https://4cawmi2va33i3w6dek1d7y1m-wpengine.netdna-ssl.com/wp-content/uploads/2017/02/2_1_Full_Intro-to-Data-Science-e1486482232872-776x1024.jpg) [The Data Analysis Epicycle](https://gohighbrow.com/the-data-analysis-epicycle/)  <!-- .element: class="figcaption" -->

***

#### Six Types of Analyses

- Descriptive
- Exploratory
- Inferential
- Predictive
- Causal
- Mechanistic

<div style="float: right;width: 70%;">
[Course Introduction - Advanced Data Science by Jeff Leek](http://jtleek.com/advdatasci/slides/01-introduction-slides.html#38)  <!-- .element: class="figcaption" -->
<iframe src="http://jtleek.com/advdatasci/slides/01-introduction-slides.html#38" />
</div>

Note:

https://bookdown.org/rdpeng/artofdatascience/types-of-questions.html
https://datascientistinsights.com/2013/01/29/six-types-of-analyses-every-data-scientist-should-know/
https://www.coursera.org/specializations/jhu-data-science
https://www.pinterest.com.au/pin/572520171368225895/
http://www.caperay.com/blog/index.php/2015/ask-the-right-questions/

***

<!-- .slide: style="font-size: 16px" -->

#### Confusion Matrix <!-- .element: style="font-size: 42px" -->

<table class="wikitable" align="center" style="text-align:center; border:none; background:transparent;color: black">
<tbody><tr>
<td style="border:none;" colspan="2"></td>
<td style="background:#eeeebb;" colspan="2"><b>True condition</b></td>
</tr>
<tr>
<td style="border:none;"></td>
<td style="background:#dddddd;"><a href="https://en.wikipedia.org/wiki/Statistical_population" title="Statistical population">Total population</a></td>
<td style="background:#ffffcc;">Condition positive</td>
<td style="background:#ddddaa;">Condition negative</td>
<td style="background:#eeeecc;font-size:90%;"><a href="https://en.wikipedia.org/wiki//Prevalence" title="Prevalence">Prevalence</a> <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ&nbsp;Condition positive</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Total population</span></span></span></td>
<td style="background:#cceecc;border-left:double silver;font-size:90%;" colspan="2"><a href="https://en.wikipedia.org/wiki/Accuracy_and_precision" title="Accuracy and precision">Accuracy</a> (ACC) = <span style="font-size:118%;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ&nbsp;True positive + Σ True negative</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Total population</span></span></span></td>
</tr>
<tr>
<td style="background:#bbeeee;" rowspan="2"><b>Predicted<br>
condition</b></td>
<td style="background:#ccffff;">Predicted condition<br>
positive</td>
<td style="background:#ccffcc;"><span style="color:#006600;"><b><a href="https://en.wikipedia.org/wiki/True_positive" class="mw-redirect" title="True positive">True positive</a></b>,<br>
<a href="https://en.wikipedia.org/wiki/Statistical_power" title="Statistical power">Power</a></span></td>
<td style="background:#eedddd;"><span style="color:#cc0000;"><b><a href="https://en.wikipedia.org/wiki/False_positive" class="mw-redirect" title="False positive">False positive</a></b>,<br>
<a href="https://en.wikipedia.org/wiki/Type_I_error" class="mw-redirect" title="Type I error">Type I error</a></span></td>
<td style="background:#ccffee;border-top:double silver;font-size:90%;"><a href="https://en.wikipedia.org/wiki/Positive_predictive_value" class="mw-redirect" title="Positive predictive value">Positive predictive value</a> (PPV), <a href="https://en.wikipedia.org/wiki/Precision_(information_retrieval)" class="mw-redirect" title="Precision (information retrieval)">Precision</a> = <span style="font-size:118%;white-space:nowrap;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ True positive</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Predicted&nbsp;condition&nbsp;positive</span></span></span></td>
<td style="background:#cceeff;border-top:double silver;font-size:90%;" colspan="2"><a href="https://en.wikipedia.org/wiki/False_discovery_rate" title="False discovery rate">False discovery rate</a> (FDR) = <span style="font-size:118%;white-space:nowrap;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ False positive</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Predicted&nbsp;condition&nbsp;positive</span></span></span></td>
</tr>
<tr>
<td style="background:#aadddd;">Predicted condition<br>
negative</td>
<td style="background:#ffdddd;"><span style="color:#cc0000;"><b><a href="https://en.wikipedia.org/wiki/False_negative" class="mw-redirect" title="False negative">False negative</a></b>,<br>
<a href="https://en.wikipedia.org/wiki/Type_II_error" class="mw-redirect" title="Type II error">Type II error</a></span></td>
<td style="background:#bbeebb;"><span style="color:#006600;"><b><a href="https://en.wikipedia.org/wiki/True_negative" class="mw-redirect" title="True negative">True negative</a></b></span></td>
<td style="background:#eeddee;border-bottom:double silver;font-size:90%;"><a href="https://en.wikipedia.org/wiki/False_omission_rate" class="mw-redirect" title="False omission rate">False omission rate</a> (FOR) = <span style="font-size:118%;white-space:nowrap;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ False negative</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Predicted&nbsp;condition&nbsp;negative</span></span></span></td>
<td style="background:#aaddcc;border-bottom:double silver;font-size:90%;" colspan="2"><a href="https://en.wikipedia.org/wiki/Negative_predictive_value" class="mw-redirect" title="Negative predictive value">Negative predictive value</a> (NPV) = <span style="font-size:118%;white-space:nowrap;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ True negative</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Predicted&nbsp;condition&nbsp;negative</span></span></span></td>
</tr>
<tr style="font-size:90%;">
<td style="border:none;vertical-align:bottom;padding:0 2px 0 0;color:#999999;" colspan="2" rowspan="2">Click thumbnail for interactive chart:
<div class="noresize" style="height: 105px; width: 210px; "><a href="http://upload.wikimedia.org/wikipedia/commons/0/0e/DiagnosticTesting_Diagram.svg" class="plainlinks" rel="nofollow" title="Click for interactive SVG chart"><img alt="DiagnosticTesting Diagram.svg" src="//upload.wikimedia.org/wikipedia/commons/thumb/0/0e/DiagnosticTesting_Diagram.svg/210px-DiagnosticTesting_Diagram.svg.png" width="210" height="105" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/0/0e/DiagnosticTesting_Diagram.svg/315px-DiagnosticTesting_Diagram.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/0/0e/DiagnosticTesting_Diagram.svg/420px-DiagnosticTesting_Diagram.svg.png 2x" data-file-width="512" data-file-height="256"></a>
</div>
</td>
<td style="background:#eeffcc;"><a href="https://en.wikipedia.org/wiki/True_positive_rate" class="mw-redirect" title="True positive rate">True positive rate</a> (TPR), <a href="https://en.wikipedia.org/wiki/Recall_(information_retrieval)" class="mw-redirect" title="Recall (information retrieval)">Recall</a>, <a href="https://en.wikipedia.org/wiki/Sensitivity_(tests)" class="mw-redirect" title="Sensitivity (tests)">Sensitivity</a>, probability&nbsp;of&nbsp;detection <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ True positive</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Condition&nbsp;positive</span></span></span></td>
<td style="background:#eeddbb;"><a href="https://en.wikipedia.org/wiki/False_positive_rate" title="False positive rate">False positive rate</a> (FPR), <a href="https://en.wikipedia.org/wiki/Information_retrieval" title="Information retrieval"><span class="nowrap">Fall-out</span></a>, probability&nbsp;of&nbsp;false&nbsp;alarm <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ False positive</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Condition&nbsp;negative</span></span></span></td>
<td style="background:#eeeeee;"><a href="https://en.wikipedia.org/wiki/Positive_likelihood_ratio" class="mw-redirect" title="Positive likelihood ratio">Positive likelihood ratio</a> <span class="nowrap">(LR+)</span> <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">TPR</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">FPR</span></span></span></td>
<td style="background:#dddddd;" rowspan="2"><a href="https://en.wikipedia.org/wiki/Diagnostic_odds_ratio" title="Diagnostic odds ratio">Diagnostic odds ratio</a> (DOR) <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">LR+</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">LR−</span></span></span></td>
<td style="background:#ddffdd;border-left:double silver;line-height:2;" rowspan="2"><a href="https://en.wikipedia.org/wiki/F1_score" title="F1 score">F<sub>1</sub> score</a> = <span style="font-size:118%;white-space:nowrap;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">2</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;"><span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">1</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Recall</span></span>&nbsp;+&nbsp;<span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">1</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Precision</span></span></span></span></span></td>
</tr>
<tr style="font-size:90%;">
<td style="background:#ffeecc;"><a href="https://en.wikipedia.org/wiki/False_negative_rate" class="mw-redirect" title="False negative rate">False negative rate</a> (FNR), Miss&nbsp;rate <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ False negative</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Condition&nbsp;positive</span></span></span></td>
<td style="background:#ddeebb;"><a href="https://en.wikipedia.org/wiki/True_negative_rate" class="mw-redirect" title="True negative rate">True negative rate</a> (TNR), <a href="https://en.wikipedia.org/wiki/Specificity_(tests)" class="mw-redirect" title="Specificity (tests)">Specificity</a> (SPC) <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">Σ True negative</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">Σ&nbsp;Condition&nbsp;negative</span></span></span></td>
<td style="background:#cccccc;"><a href="https://en.wikipedia.org/wiki/Negative_likelihood_ratio" class="mw-redirect" title="Negative likelihood ratio">Negative likelihood ratio</a> <span class="nowrap">(LR−)</span> <span style="font-size:118%;white-space:nowrap;">= <span class="sfrac nowrap" style="display:inline-block; vertical-align:-0.5em; font-size:85%; text-align:center;"><span style="display:block; line-height:1em; margin:0 0.1em;">FNR</span><span class="visualhide">/</span><span style="display:block; line-height:1em; margin:0 0.1em; border-top:1px solid;">TNR</span></span></span></td>
</tr>
</tbody></table>

[Wikipedia: Sensitivity and specificity](https://en.wikipedia.org/wiki/Sensitivity_and_specificity) <!-- .element: class="figcaption" -->

***

<!-- .slide: style="font-size: 32px" -->

#### Confusion Matrix (Simplified)

|  Term    |            Formula                    |    Focus     |
|----------|---------------------------------------|--------------|
| Accuracy | (True Positive + True Negative) / All |  Right       |
| Precision | True Positive / All Positive | Right about Positive |
| Sensitivity/Recall | True Positive / All Real (= True Positive + False Negative) | Right about Real |
| Specificity | True Negative / All NOT Real (= True Negative + False Positive)  | Right about NOT Real |

---

## Traditional Machine Learning Methods

Note:

http://detexify.kirelabs.org/classify.html
http://shapecatcher.com/
https://math.meta.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference

https://towardsdatascience.com/machine-learning-quick-reference-card-cf92f6accd08
[Essentials of Machine Learning Algorithms with Python and R Codes](https://www.analyticsvidhya.com/blog/2017/09/common-machine-learning-algorithms/)
[ The 10 Algorithms Machine Learning Engineers Need to Know](https://www.kdnuggets.com/2016/08/10-algorithms-machine-learning-engineers.html)

---

### Linear Regression

***

#### Elements of Linear Relationships

![](https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/ch10/images/10_01.png)  <!-- .element: class="img-450" --> [Python Machine Learning: Chapter 10 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch10/ch10.ipynb)  <!-- .element: class="figcaption" -->

Note:

https://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression/

[Machine Learning FAQ by Sebastian Raschka](https://sebastianraschka.com/faq/docs/closed-form-vs-gd.html)

![](https://imgs.xkcd.com/comics/linear_regression.png) [xkcd: Linear Regression](https://xkcd.com/1725/)  <!-- .element: class="figcaption" -->

***

#### Assumption `$$ y_i = \beta_0 + \beta x_i + \varepsilon $$`

![](https://image.slidesharecdn.com/simplelinearregressionpelatihan-090829234643-phpapp02/95/simple-linier-regression-9-728.jpg?cb=1251589640)   <!-- .element: class="img-450" --> [ by Department of Statistics, ITS Surabaya](https://www.slideshare.net/dessybudiyanti/simple-linier-regression)  <!-- .element: class="figcaption" -->

***

#### Linear Model in Matrix Form

`$$ \boldsymbol{\hat{y}} = X \boldsymbol{w} + \boldsymbol{b} $$`

For $p$ features and $n$ samples: <!-- .element: style="font-size: 24px" -->

`$$ \left(
  \begin{array}{c}
    \hat{y}_1 \\
    \vdots \\
    \hat{y}_n
  \end{array}
\right)
= \left(
  \begin{array}{c}
    \mathbf{x}^\top_1 \\
    \vdots \\
    \mathbf{x}^\top_n
  \end{array}
\right) \left(
  \begin{array}{c}
    w_1 \\
    \vdots \\
    w_p
  \end{array}
\right) + b $$` <!-- .element: class="" style="font-size: smaller" --> 

`$$ = \left(
  \begin{array}{cccc}
    1 & x_{11} & \ldots & x_{1p}\\
    1 & \vdots & \ddots & \vdots\\
    1 & x_{n1} & \ldots & x_{np}
  \end{array}
\right) \left(
  \begin{array}{c}
    w_0 \\
    \vdots \\
    w_p
  \end{array}
\right) $$` <!-- .element: class="fragment" style="font-size: smaller" -->

Note:

https://en.wikipedia.org/wiki/Linear_regression

***

#### Bias trick

![](http://cs231n.github.io/assets/wb.jpeg) [CS231n Convolutional Neural Networks for Visual Recognition: Linear classification](http://cs231n.github.io/linear-classify/)  <!-- .element: class="figcaption" -->

***

### The Objective

![minimal](https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Extrema_example_original.svg/330px-Extrema_example_original.svg.png)

`$$ \underset{x \in D}{\operatorname{arg\,min}} \, f(x) := \{x \mid \forall y \in D : f(y) \ge f(x)\} $$`

Note:

- https://en.wikipedia.org/wiki/Arg_max
- [Argmax and Max Calculus](https://www.cs.ubc.ca/~schmidtm/Documents/2016_540_Argmax.pdf) 

***

#### Ordinary Least Squares

![](http://gluon.mxnet.io/_images/linear-regression.png) [Linear regression from scratch](http://gluon.mxnet.io/chapter02_supervised-learning/linear-regression-scratch.html) <!-- .element: class="figcaption" -->

`$$
\underset{\boldsymbol{w}}{\operatorname{arg\,min}} \, \xi(\hat{y}, y) = \ell_2(\hat{y}, y) \\
= {|| \hat{y} - y||_2}^2 = {|| X \boldsymbol{w} - \boldsymbol{y}||_2}^2 $$`

Note:

https://image.slidesharecdn.com/simplelinearregressionfinal-121101072438-phpapp02/95/simple-linear-regression-final-8-638.jpg?cb=1507030484

https://www.slideshare.net/harshupadhyay/simple-linear-regression-final?next_slideshow=1

https://www.jianshu.com/p/f71848c7aaf3

Terms related to OLS

- Ordinary Least Squares (OLS)
- Least Squares Estimator (LSE)
- Sum of Squared Errors (SSE)
- Mean Squared Error (MSE)

[Lecture Slides (Machine Learning, CS 6375)](http://www.hlt.utdallas.edu/~vgogate/ml/2015s/lectures.html)

***

#### Empirical Risk Minimization (ERM)

![](images/10701/erm.png) <!-- .element: class="img-500" --> [Machine Learning 10-701/15-781, Fall 2010 Lecture 6: Linear Regression](http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture6.pdf) <!-- .element: class="figcaption" -->

***

#### Bayes’ Theorem

![](https://www.probabilisticworld.com/wp-content/uploads/2016/06/bayes-theorem-with-description-2.png) <!-- .element: style="width: 50%" --> [Frequentist And Bayesian Approaches In Statistics](https://www.probabilisticworld.com/frequentist-bayesian-approaches-inferential-statistics/) <!-- .element: class="figcaption" -->

![](https://www.probabilisticworld.com/wp-content/uploads/2016/03/coin-bias-posterior.png) <!-- .element: style="width: 50%" --> [Calculating Coin Bias With Bayes’ Theorem](https://www.probabilisticworld.com/calculating-coin-bias-bayes-theorem/) <!-- .element: class="figcaption" -->

Note:

http://gandenberger.org/2014/07/28/intro-to-statistical-methods-2/
![](http://gandenberger.org/wp-content/uploads/2014/07/Statistical_Methods.png)

https://medium.com/data-science-group-iitr/naive-bayes-unfolded-b2ab036b42b1
https://www.explainxkcd.com/wiki/index.php/795:_Conditional_Risk
https://betterexplained.com/articles/an-intuitive-and-short-explanation-of-bayes-theorem/

https://qph.ec.quoracdn.net/main-qimg-18b90ce3920e0913240c0b5b584e4d19.webp
https://www.quora.com/Do-extraordinary-claims-require-extraordinary-evidence

http://www.labtimes.org/labtimes/method/methods/img/2011_04b.jpg
http://www.labtimes.org/labtimes/method/methods/2011_04.lasso

***

#### MLE v.s. MAP

![](images/10701/mle_vs_map.png) <!-- .element: class="img-500" --> [Machine Learning 10-701/15-781, Fall 2010 Lecture 2](http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture2.pdf) <!-- .element: class="figcaption" -->

Maximum Likelihood Estimation v.s. Maximum A Posteriori Estimation  <!-- .element: style="font-size: 24px" -->

***

#### M(C)LE v.s. M(C)AP

![](images/10701/mcle_vs_mcap.png) <!-- .element: class="img-500" --> [Machine Learning 10-701/15-781, Fall 2010 Lecture 2](http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture2.pdf) <!-- .element: class="figcaption" -->

***

#### Least Squares and MLE

![](images/10701/lse_mle.png) <!-- .element: class="img-500" --> [Machine Learning 10-701/15-781, Fall 2010 Lecture 6: Linear Regression](http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture6.pdf) <!-- .element: class="figcaption" -->

Least Square Estimate is same as Maximum Likelihood Estimate under a Gaussian model. <!-- .element: style="font-size: 24px" -->

Note:

Maximum A Posteriori Estimation

http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture2.pdf

***

#### Implementing Linear Regression

<!-- .slide: style="font-size: 32px;" -->

```python
def nn(x, w):
    return x * w
def cost(y, t):
    return ((y - t)**2).sum()

nb_of_samples = 20
x = np.random.uniform(0, 1, nb_of_samples)
t = 3 * x + np.random.normal(0, 0.2, nb_of_samples)

# And then...
```

<p class="fragment current-only" data-code-focus="1-2">
  `$ \boldsymbol{\hat{y}} = X \boldsymbol{w} $`
</p>
<p class="fragment current-only" data-code-focus="3-4">
  `$ \underset{\boldsymbol{w}}{\operatorname{arg\,min}} \, \xi(\hat{y}, t) = {|| \hat{y} - t||_2}^2 $`
</p>
<p class="fragment current-only" data-code-focus="6-8" style="font-size: 24px">
  `$$
    x \sim {\mathcal {U}}(0, 1) \\
    t = 3 x + \varepsilon \\
    \varepsilon \sim N(\mu, \sigma^2) \; \text{with} \; \mu = 0, \sigma = 0.2
  $$`
</p>
<p class="fragment current-only" data-code-focus="10" style="">
  How to optimize?
</p>
---

### Optimization Methods

***

#### Coordinate Descent

![](http://hduongtrong.github.io/assets/gradient_descent/coordinate_descent.gif) <!-- .element: style="height: 300px; background-color: white" --> [Gradient Descent and Variants - Convergence Rate Summary](http://hduongtrong.github.io/2015/11/23/coordinate-descent/)  <!-- .element: class="figcaption" -->

`$$
  w_{k+1} \gets w_k - \alpha_k \nabla_{i_k} F(w_k) e_{i_k}
$$`

`$$
  \ \ \text{where}\ \ \nabla_{i_k} F(w_k) := \frac{\partial F}{\partial w^{i_k}}(w_k)
$$` <!-- .element: class="fragment" style="font-size: smaller" -->

Note:

`$$ \underset{w}{\operatorname{arg\,min}} \, F : \mathbb{R}^{d} \to \mathbb{R} $$`

$w^{i_k}$ represents the $i_k$-th element of the parameter vector, and $e_{i_k}$ represents the $i_k$-th coordinate vector for some $i_k \in \{1,\dots,d\}$.  In other words, the solution estimates $w_{k+1}$ and $w_k$ differ only in their $i_k$-th element as a result of a move in the $i_k$-th coordinate from $w_k$.

***

#### Gradient Descent (GD)

![](https://sebastianraschka.com/images/faq/closed-form-vs-gd/ball.png)   <!-- .element: class="img-300" --> [Machine Learning FAQ by Sebastian Raschka](https://sebastianraschka.com/faq/docs/closed-form-vs-gd.html)   <!-- .element: class="figcaption" -->

`$$
\text{Stochastic: }\quad w_{k+1} \gets w_k - \alpha_k \nabla f_{i_k}(w_k)
$$`

`$$
\text{Batch: }\quad  w_{k+1} \gets w_k - \frac{\alpha_k}{n} \sum_{i=1}^n \nabla f_i(w_k)
$$`

***

#### Implementing Linear Regression with GD

<!-- .slide: style="font-size: 32px;" -->

```python
def nn(x, w): return x * w
def cost(y, t): return ((y - t)**2).sum()
nb_of_samples = 20
x = np.random.uniform(0, 1, nb_of_samples)
t = 3 * x + np.random.normal(0, 0.2, nb_of_samples)

def gradient(w, x, t): return 2 * (nn(x, w) - t) * x
def delta_w(w, x, t, learning_rate):
    return learning_rate * gradient(w, x, t).sum()

w = 0.1; learning_rate = 0.1; nb_of_iterations = 10;
w_cost = [(w, cost(nn(x, w), t))]
for i in range(nb_of_iterations):
    w = w - delta_w(w, x, t, learning_rate)
    w_cost.append((w, cost(nn(x, w), t)))
```

<p class="fragment current-only" data-code-focus="1">
  `$ \boldsymbol{\hat{y}} = X \boldsymbol{w} $`
</p>
<p class="fragment current-only" data-code-focus="2">
  `$ \underset{\boldsymbol{w}}{\operatorname{arg\,min}} \, \xi(\hat{y}, t) = {|| \hat{y} - t||_2}^2 $`
</p>
<p class="fragment current-only" data-code-focus="3-5" style="font-size: 24px">
  `$$
    x \sim {\mathcal {U}}(0, 1) \\
    t = 3 x + \varepsilon \\
    \varepsilon \sim N(\mu, \sigma^2) \; \text{with} \; \mu = 0, \sigma = 0.2
  $$`
</p>
<p class="fragment current-only" data-code-focus="7" style="font-size: 24px">
  `$$
      \frac{\partial \xi(\hat{y}, t)}{\partial w} = \frac{\partial \xi}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial w}
    = \frac{\partial (\hat{y} - t)^2}{\partial \hat{y}} \frac{\partial (x w)}{\partial w} \\
    = - 2 (\hat{y} - t) x = - 2 (x w - t) x
  $$`
</p>
<p class="fragment current-only" data-code-focus="8-9" style="font-size: 24px">
  `$$
    \Delta w = \mu \sum_{i=1}^{N} \frac{\partial \xi_i(\hat{y}_i, t_i)}{\partial w}
  $$`
</p>
<p class="fragment current-only" data-code-focus="11-15" style="">
  `$$
    w_{k+1} \gets w_k - \alpha_k \Delta w
  $$`
</p>

***

<!-- .slide: data-background-iframe="http://nbviewer.jupyter.org/github/peterroelants/peterroelants.github.io/blob/master/notebooks/neural_net_implementation/neural_network_implementation_part01.ipynb" data-background-interactive -->

***

#### Beyond SGD: Noise Reduction and Second-Order Methods

![](images/beyond_sgd.png) <!-- .element: style="height: 300px" -->

[Optimization Methods for Large-Scale Machine Learning](https://arxiv.org/pdf/1606.04838.pdf) <!-- .element: class="figcaption" -->

***

#### GD Optimization Algorithms: Noisy Moons

![](http://hduongtrong.github.io/assets/gradient_descent/all.gif) [Alec Radford](https://www.reddit.com/r/MachineLearning/comments/2gopfa/visualizing_gradient_optimization_techniques/cklhott/
)  <!-- .element: class="figcaption" -->

Note:

Comparison of a few optimization methods (animation by Alec Radford). The star denotes the global minimum on the error surface. Notice that stochastic gradient descent (SGD) without momentum is the slowest method to converge in this example.

[Using convolutional neural nets to detect facial keypoints tutorial](http://danielnouri.org/notes/category/machine-learning/#testing-it-out)
[Gradient Descent and Variants - Convergence Rate Summary](http://hduongtrong.github.io/2015/11/23/coordinate-descent/)
[**Test functions for optimization**](https://en.wikipedia.org/wiki/Test_functions_for_optimization)

***

#### GD Optimization Algorithms: Long Valley

![](https://i.imgur.com/2dKCQHh.gif) [Alec Radford: Visualizing Optimization Algorithms](https://imgur.com/a/Hqolp) <!-- .element: class="figcaption" -->

Note:

Algos without scaling based on gradient information really struggle to break symmetry here - SGD gets no where and Nesterov Accelerated Gradient / Momentum exhibits oscillations until they build up velocity in the optimization direction.

Algos that scale step size based on the gradient quickly break symmetry and begin descent.

http://ruder.io/optimizing-gradient-descent/
http://cs231n.github.io/neural-networks-3/
http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM
https://github.com/robertsdionne/bouncingball
http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html

***

#### GD Optimization Algorithms: Beale's function

![](https://i.imgur.com/pD0hWu5.gif) [Alec Radford: Visualizing Optimization Algorithms](https://imgur.com/a/Hqolp) <!-- .element: class="figcaption" -->

Note:

Due to the large initial gradient, velocity based techniques shoot off and bounce around - adagrad almost goes unstable for the same reason.

Algos that scale gradients/step sizes like adadelta and RMSProp proceed more like accelerated SGD and handle large gradients with more stability.

https://phyblas.hinaboshi.com/rup/nayuki/2017/e15.gif

***

#### GD Optimization Algorithms: Saddle Point

![](https://i.imgur.com/NKsFHJb.gif) [Alec Radford: Visualizing Optimization Algorithms](https://imgur.com/a/Hqolp) <!-- .element: class="figcaption" -->

Note:

Behavior around a saddle point.

NAG/Momentum again like to explore around, almost taking a different path. 

Adadelta/Adagrad/RMSProp proceed like accelerated SGD.

https://phyblas.hinaboshi.com/rup/nayuki/2017/e16.gif

https://nathanbrixius.wordpress.com/2016/07/29/stochastic-and-batch-methods-for-machine-learning/
http://bair.berkeley.edu/blog/2017/08/31/saddle-efficiency/
https://www.slideshare.net/diannepatricia/martin-takac-solving-largescale-machine-learning-problems-in-a-distributed-way
http://andrew.gibiansky.com/blog/machine-learning/hessian-free-optimization/
http://www.cs.toronto.edu/~jmartens/docs/Momentum_Deep.pdf
http://runopti.github.io/blog/2016/07/07/HessianComp/
http://scikit-learn.org/stable/modules/sgd.html

***

<!-- .slide: data-background-iframe="https://distill.pub/2017/momentum/" data-background-interactive -->

***

#### Nesterov Momentum

![](http://cs231n.github.io/assets/nn3/nesterov.jpeg) [CS231n Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/)  <!-- .element: class="figcaption" -->

***

#### Adam = SDG + Adaptive + Momentum

* [SGD, SGD-M, NAG, AdaGrad, AdaDelta/RMSProp, Adam, Nadam](https://zhuanlan.zhihu.com/p/32230623)
* [Why not just use Adam?](https://zhuanlan.zhihu.com/p/32262540)
* [Adam -> SGD](https://zhuanlan.zhihu.com/p/32338983)

---

### Regulations

***

#### Overfitting

![](images/overfitting.gif)

[Over-fitting and Regularization](https://towardsdatascience.com/over-fitting-and-regularization-64d16100f45c)   <!-- .element: class="figcaption" -->

***

#### Outliers: Anscombe's quartet

![Anscombe's quartet](https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/638px-Anscombe%27s_quartet_3.svg.png) [Linear regression From Wikipedia](https://en.wikipedia.org/wiki/Linear_regression) <!-- .element: class="figcaption" -->

***

#### Find Good Compromise

![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch03/images/03_06.png?raw=true) [Python Machine Learning: Chapter 3 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch03/ch03.ipynb)  <!-- .element: class="figcaption" -->

***

#### Vector Norms

`$$ ||x||_p=(\sum_i |x_i|^p)^{1/p} $$`

***

#### Typical Vector Norms

|                 |               |                                      |
|-----------------|---------------|--------------------------------------|
| `$\ell_0$`-Norm | `$ ∣∣x∣∣_0 $` | `$ \#(i \mid x_i \neq 0) $` |
| `$\ell_1$`-Norm | `$ ∣∣x∣∣_1 $` | `$ \sum_i ∣x_i∣ $` |
| `$\ell_2$`-Norm | `$ ∣∣x∣∣_2 $` | `$ \sqrt{\sum_i^{\phantom{n}} ∣x_i∣^2} $` |
| `$\ell_\infty$`-Norm | `$ ∣∣x∣∣_\infty $` | `$ \max_i ∣x_i∣ $` |

[l0-Norm, l1-Norm, l2-Norm, … , l-infinity Norm](https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/)  <!-- .element: class="figcaption see-also" -->

***

#### Typical Vector Norms

![](http://www.stat.ucla.edu/~ybzhao/teaching/stat101c/im/lpball.png)

[Introduction to Statistical Models and Data Mining
](http://www.stat.ucla.edu/~ybzhao/teaching/stat101c/) <!-- .element: class="figcaption" -->

Note:

https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization-How-does-it-solve-the-problem-of-overfitting-Which-regularizer-to-use-and-when
http://blog.shakirm.com/2016/04/learning-in-brains-and-machines-2/
https://cnx.org/contents/U4hLPGQD@5/Compressible-signals#uid10

***

#### Regulations

<!-- .slide: style="font-size:smaller" -->

- Ridge: `$$ \underset{w}{min\,} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2} $$`
- Lasso: `$$ \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha ||w||_1} $$`
- Elastic Net:  `$$ \underset{w}{min\,} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 + \frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2} $$`

***

#### Sparsity

| Ridge                                  | Lasso                               |
|----------------------------------------|-------------------------------------|
| ![](https://rasbt.github.io/mlxtend/user_guide/general_concepts/regularization-linear_files/l2.png) | ![](https://rasbt.github.io/mlxtend/user_guide/general_concepts/regularization-linear_files/l1.png)  |

[Regularization of Generalized Linear Models by Sebastian Raschka](https://rasbt.github.io/mlxtend/user_guide/general_concepts/regularization-linear/) <!-- .element: class="figcaption" -->

Note:

![](images/10701/l0_l1_l2.png)  <!-- .element: style="height: 500px" --> [Machine Learning 10-701/15-781, Fall 2010 Lecture 6: Linear Regression](http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture6.pdf) <!-- .element: class="figcaption" -->

[What are the main regularization methods used in machine learning? -- Somdeb Sarkhel's answer](https://www.quora.com/What-are-the-main-regularization-methods-used-in-machine-learning/answer/Somdeb-Sarkhel?srid=xQkR)  <!-- .element: class="figcaption see-also" -->

https://stats.stackexchange.com/questions/45643/why-l1-norm-for-sparse-models

***

#### Prior

| Ridge: Gaussian                        | Lasso: Laplace                      |
|----------------------------------------|-------------------------------------|
| ![](images/10701/map_ridge.png)        |![](images/10701/map_lasso.png)      |

[Machine Learning 10-701/15-781, Fall 2010 Lecture 6: Linear Regression](http://www.cs.cmu.edu/~aarti/Class/10701/slides/Lecture6.pdf) <!-- .element: class="figcaption" -->

Note:

https://towardsdatascience.com/the-surprising-longevity-of-the-z-score-a8d4f65f64a0
http://xaktly.com/ProbStat_Distributions.html

---

### Logistic Regression

***

#### Logistic Regression is Linear Classification

![](images/lr.png) [Logistic Regression – Geometric Intuition](https://florianhartl.com/logistic-regression-geometric-intuition.html)   <!-- .element: class="figcaption" -->

***

#### Logistic Regression is Linear Classification

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~utensil/1.embed"></iframe>

[Logistic Regression – Geometric Intuition](https://florianhartl.com/logistic-regression-geometric-intuition.html)   <!-- .element: class="figcaption" -->

***

#### Logistic Regression with 2 Features

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~utensil/3.embed"></iframe>

[Logistic Regression – Geometric Intuition](https://florianhartl.com/logistic-regression-geometric-intuition.html)   <!-- .element: class="figcaption" -->

***

<!-- .slide: style="font-size:smaller" -->

#### Linear Regression v.s. Logistic Regression

![](https://cdn-images-1.medium.com/max/1000/1*Oy6O6OdzTXbp_Czi_k4mRg.png) [Gentlest Intro to Tensorflow #4: Logistic Regression](https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-4-logistic-regression-2afd0cabc54)  <!-- .element: class="figcaption" -->

***

#### Linear Regression v.s. Logistic Regression (Cont.)

![](https://cdn-images-1.medium.com/max/1000/1*1L4lmdLXtcfXbS6urKOKrw.png) [Gentlest Intro to Tensorflow #4: Logistic Regression](https://medium.com/all-of-us-are-belong-to-machines/gentlest-intro-to-tensorflow-4-logistic-regression-2afd0cabc54) <!-- .element: class="figcaption" -->

***

#### Schematic of Logistic Regression

![](http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression_files/logistic_regression_schematic.png) [Logistic Regression by Sebastian Raschka](http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression/)  <!-- .element: class="figcaption" -->

***

#### Logistic Function ( a.k.a Sigmoid )

![](https://isaacchanghau.github.io/images/deeplearning/activationfunction/sigmoid.png) <!-- .element: class="img-300" --> [Activation Functions in Artificial Neural Networks](https://isaacchanghau.github.io/2017/05/22/Activation-Functions-in-Artificial-Neural-Networks/)  <!-- .element: class="figcaption" -->

`$$ \sigma(x) = \frac{1}{1+e^{-x}} $$`   <!-- .element: class="fragment current-only" -->

`$$ \hat{y} = \sigma(X \boldsymbol{w}) = \frac{1}{1+e^{-X \boldsymbol{w}}} $$`   <!-- .element: class="fragment current-only" -->

`$$ \hat{y} = \sigma(X \boldsymbol{w}) = \frac{1}{2}(\tanh(X \boldsymbol{w}) + 1)$$`   <!-- .element: class="fragment current-only" -->

Note:

https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons
https://www.kdnuggets.com/2016/08/role-activation-function-neural-network.html
https://en.wikipedia.org/wiki/Activation_function

***

#### Hyperbolic Tangent (tanh)

![](https://isaacchanghau.github.io/images/deeplearning/activationfunction/tanh.png)   <!-- .element: class="img-300" -->  [Activation Functions in Artificial Neural Networks](https://isaacchanghau.github.io/2017/05/22/Activation-Functions-in-Artificial-Neural-Networks/)  <!-- .element: class="figcaption" -->

`$$
\tanh(x) = \frac{1-e^{-2x}}{1+e^{-2x}} = 2\sigma (2x)-1 
$$`

`$$
\tanh(x) \in (-1, 1) \; \text{while} \; \sigma (x) \in (0,1)
$$`  <!-- .element: class="fragment" -->

***

#### Binary Cross-Entropy Loss

![](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/img/1d8fc59e6a674f1c.png)   <!-- .element: class="img-300" -->  [TensorFlow and deep learning, without a PhD
](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#4)  <!-- .element: class="figcaption" -->

`$$ \xi(y,\hat{y}) = - \sum_{i=1}^{n} \left[ y_i log(\hat{y}_i) + (1-y_i)log(1-\hat{y}_i) \right] $$`

***

<!-- .slide: style="font-size:smaller" -->

#### Impelementing Logistic Regression

```python
def logistic(z): 
    return 1 / (1 + np.exp(-z))

def nn(x, w): 
    return logistic(x.dot(w.T))

def nn_predict(x,w): 
    return np.around(nn(x,w))

def cost(t, y):
    return - np.sum(
      np.multiply(t, np.log(y)) + 
      np.multiply((1-t), np.log(1-y))
    )
```

<p class="fragment current-only" data-code-focus="1-2">
`$$ \sigma(z) = \frac{1}{1+e^{-z}} $$`
</p>
<p class="fragment current-only" data-code-focus="4-5">
`$$ y = \sigma(X \boldsymbol{w}) $$`
</p>
<p class="fragment current-only" data-code-focus="7-8">
`$$ \hat{y} = \left\lfloor y + 0.5 \right\rfloor = -\left\lceil -y - 0.5 \right\rceil $$`

[Wikipedia: Rounding](https://en.wikipedia.org/wiki/Rounding)  <!-- .element: class="figcaption" -->
</p>
<p class="fragment current-only" data-code-focus="10-14">
`$$ \xi(t,\hat{y}) = - \sum_{i=1}^{n} \left[ t_i log(\hat{y}_i) + (1-t_i)log(1-\hat{y}_i) \right] $$`
</p>

***

<!-- .slide: data-background-iframe="http://nbviewer.jupyter.org/github/peterroelants/peterroelants.github.io/blob/master/notebooks/neural_net_implementation/neural_network_implementation_part02.ipynb" data-background-interactive -->

---

### Softmax Regression

***

#### Softmax

`$$ y_c = \varsigma(\mathbf{x})_c = \frac{e^{x_c}}{\sum_{i=1}^C e^{x_i}} \quad \text{for} \; c = 1 \cdots C $$`

where `$ \mathbf{x} , y \in \mathbb{R}^C, y_c \in [0, 1], \sum\limits_{c=1}^C y_c = 1$`

Softmax is a soft version of arg max :  <!-- .element: class="fragment" -->

`$$ \operatorname{arg\,max} ([3, 5, 0]) = [0, 1, 0] $$`  <!-- .element: class="fragment" -->

`$$ \operatorname{softmax} ([3, 5, 0]) \approx [0.12, 0.88, 0] $$`   <!-- .element: class="fragment current-only" -->

`$$ \operatorname{soft\,arg\,max} ([3, 5, 0]) \approx [0.12, 0.88, 0] $$`   <!-- .element: class="fragment current-only" -->

Note:

http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/
https://www.quora.com/Why-is-softmax-activate-function-called-softmax

***

#### Softmax Predicts Probability

![](https://isaacchanghau.github.io/images/deeplearning/activationfunction/softmax.png)  <!-- .element: class="img-450" -->  [Activation Functions in Artificial Neural Networks](https://isaacchanghau.github.io/2017/05/22/Activation-Functions-in-Artificial-Neural-Networks/)  <!-- .element: class="figcaption" -->

***

#### Schematic of Softmax Regression

![](http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression_files/softmax_schematic_1.png)  <!-- .element: class="img-450" -->  [Softmax Regression by Sebastian Raschka](http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/)  <!-- .element: class="figcaption" -->

***

#### Cross-Entropy Loss for Softmax

`$$ \underset{\theta}{\operatorname{arg\,max}}\; \mathcal{L}(\theta|\mathbf{t},\mathbf{z}) $$`

`$$ = \underset{\theta}{\operatorname{arg\,min}} \; - log \mathcal{L}(\theta|\mathbf{t},\mathbf{z}) = \xi(\mathbf{t},\mathbf{z}) \\
= - log \prod_{i=c}^{C} y_c^{t_c} = - \sum_{i=c}^{C} t_c \cdot log(y_c) $$`

[How to implement Softmax Classification](http://peterroelants.github.io/posts/neural_network_implementation_intermezzo02/)  <!-- .element: class="figcaption" -->

***

#### Minimizing Cross-Entropy Loss Maximizes the Right Probability

![](https://image.slidesharecdn.com/gentlestintrototensorflowpart3-160629045311/95/gentlest-introduction-to-tensorflow-part-3-66-638.jpg?cb=1486266200) <!-- .element: class="img-left" -->

![](https://image.slidesharecdn.com/gentlestintrototensorflowpart3-160629045311/95/gentlest-introduction-to-tensorflow-part-3-67-638.jpg?cb=1486266200) <!-- .element: class="img-right" -->

[Gentlest Introduction to Tensorflow - Part 3](https://www.slideshare.net/KhorSoonHin/gentlest-introduction-to-tensorflow-part-3)   <!-- .element: class="figcaption" -->

Note:

![](https://image.slidesharecdn.com/gentlestintrototensorflowpart3-160629045311/95/gentlest-introduction-to-tensorflow-part-3-68-638.jpg?cb=1486266200) 

***

#### Sigmoid v.s. Softmax

Sigmoid: two-class logistic regression

`$$
\begin{align}
\Pr(Y_i=0) &= \frac{e^{-\boldsymbol\beta \cdot \mathbf{X}_i}} {1 +e^{-\boldsymbol\beta_0 \cdot \mathbf{X}_i}} \, \\
\Pr(Y_i=1) &= 1 - \Pr(Y_i=0) = \frac{1} {1 +e^{-\boldsymbol\beta \cdot \mathbf{X}_i}}
\end{align}
$$`

Softmax: multiple-class logistic regression

`$$ 
\Pr(Y_i=k) = \frac{e^{\boldsymbol\beta_k \cdot \mathbf{X}_i}} {\sum\limits_{0 \leq c \leq K} {e^{\boldsymbol\beta_c \cdot \mathbf{X}_i}}}
$$`

***

#### Implementing Softmax Regression

```python
z = [
  [30, 60, 90],
  [1, 3, 4.5]
]

def softmax_naive(z):
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

def softmax(z):
    # Avoid numerical overflow by removing max
    e = np.exp(z - np.amax(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)
```

***

#### Try it out

```python
>>> softmax_naive(z)
array([[  8.75651076e-27,   9.35762297e-14,   1.00000000e+00],
       [  2.40937683e-02,   1.78030206e-01,   7.97876026e-01]])
>>> np.argmax(softmax_naive(z))
2
>>> np.around(softmax_naive(z))
array([[ 0.,  0.,  1.],
       [ 0.,  0.,  1.]])
>>> softmax(z)
array([[  8.75651076e-27,   9.35762297e-14,   1.00000000e+00],
       [  2.40937683e-02,   1.78030206e-01,   7.97876026e-01]])
>>> np.argmax(softmax(z))
2
>>> np.around(softmax(z))
array([[ 0.,  0.,  1.],
       [ 0.,  0.,  1.]])
```
---

### SVM

***

#### Support Vector Machine (SVM)

![](https://kernelmachine.github.io/public/20170304/svm.png)  <!-- .element: style="height: 400px" -->
 [Crash Course on Support Vector Machines](https://www.suchin.co/2017/03/04/The-Support-Vector-Machine/) <!-- .element: class="figcaption" -->

Note:

<iframe width="900" height="800" frameborder="0" scrolling="no" src="//plot.ly/~utensil/5.embed"></iframe>

[Logistic Regression – Geometric Intuition](https://florianhartl.com/logistic-regression-geometric-intuition.html)   <!-- .element: class="figcaption" -->

***

#### SVM: Objective

![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch03/images/03_07.png?raw=true) [Python Machine Learning: Chapter 3 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch03/ch03.ipynb)  <!-- .element: class="figcaption" -->

Note:

The SVM amounts to finding hyperplanes such that `$ w^T_{j}x_i - w^T_{y_i}x_i  - b  \ge \delta \text{   } \forall  \text{   }  j \neq y_i
$` where `$ \delta $` is called the margin. 

***

<!-- .slide: data-background-iframe="http://vision.stanford.edu/teaching/cs231n-demos/linear-classify/" data-background-interactive -->

***

#### SVM: Hinge Loss

`$$
\min_{\mathbf{w}}\ C\underset{Hinge-Loss}{\underbrace{\sum_{i=1}^{n}max[1-y_{i}\underset{h({\mathbf{x}_i})}{\underbrace{(w^{\top}{\mathbf{x}_i}+b)}},0]}}+\underset{l_{2}-Regularizer}{\underbrace{\left\Vert w\right\Vert _{z}^{2}}}
$$`

***

#### SVM: Hinge Loss ( Another Form )

`$$
\underset{Hinge-Loss}{\underbrace{\ \sum_{i=1}^n\big(1-y_i \langle x_i,w \rangle\big)_+}} + \underset{w}{min}\ \lambda\underset{l_{2}-Regularizer}{\underbrace{\parallel w\parallel^2}}
$$`

[A Support Vector Machine in just a few Lines of Python Code](https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/)  <!-- .element: class="figcaption" -->

***

#### SVM: Hinge Loss ( The Derivatives )

`$$
\frac{\partial}{\partial w_k} \lambda\parallel w\parallel^2 \ = 2 \lambda w_k
$$`

`$$
\frac{\partial}{\partial w_k} \big(1-y_i \langle x_i,w \rangle\big)_+ \ = \begin{cases}
    0,                       & y_i \langle x_i,w \rangle\geq 1\\
    -y_ix_{ik},              & \text{otherwise}
\end{cases}
$$`

`$$
\implies \Delta w = \begin{cases}
    2\lambda w - y_ix_i,     & y_i \langle x_i,w \rangle\lt 1\\
    2\lambda w,              & \text{otherwise}
\end{cases}
$$`

[A Support Vector Machine in just a few Lines of Python Code](https://maviccprp.github.io/a-support-vector-machine-in-just-a-few-lines-of-python-code/)  <!-- .element: class="figcaption" -->

***

#### Hinge Loss is "Soft" Zero-One Loss

![](http://www.cs.cornell.edu/courses/cs4780/2015fa/web/lecturenotes/pngPic/c4/classificationlosses.png) <!-- .element: class="img-450" --> [Machine Learning CS4780/CS5780: Empirical Risk Minimization](http://www.cs.cornell.edu/courses/cs4780/2015fa/page4/index.html)  <!-- .element: class="figcaption" -->

***

#### Soft margin SVM

![](https://qph.ec.quoracdn.net/main-qimg-be899c45024583fbcce41af30e0acd98.webp)  [In layman's terms, how does SVM work?](https://www.quora.com/In-laymans-terms-how-does-SVM-work/answer/Prasoon-Goyal?srid=xQkR)  <!-- .element: class="figcaption" -->

***

#### Kernel Functions in SVM

![](http://i.imgur.com/WuxyO.png) [
Rahul Agarwal's Answer: What are kernels in machine learning and SVM and why do we need them?](https://www.quora.com/What-are-kernels-in-machine-learning-and-SVM-and-why-do-we-need-them/answer/Rahul-Agarwal-10?srid=xQkR) <!-- .element: class="figcaption" -->

Note:

https://www.quora.com/In-laymans-terms-how-does-SVM-work/answer/Prasoon-Goyal
https://www.quora.com/What-is-a-radial-basis-function
[Kernel Methods for Deep Learning](http://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf)
https://isaacchanghau.github.io/2017/08/04/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0-%E5%91%A8%E5%BF%97%E5%8D%8E-%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0-3/

***

#### Kernel Functions in SVM

- Linear: `$ K(x, y) = x^\top y $`
- Polynomial: `$ K(x, y) = (x^\top y + 1)^d $`
- Sigmoid: `$ K(x, y) = tanh(a x^\top y + b) $`
- Radial basis func(RBF): `$ K(x, y) = \exp(-\gamma \| x - y\|^2) $`

[How do I select SVM kernels?](https://www.quora.com/How-do-I-select-SVM-kernels/answer/Prasoon-Goyal)  <!-- .element: class="figcaption" -->

***

#### Impelementing SVM

```python
def nn(x, w): np.sign(np.dot(x, w)).astype(int)
def cost(y, t): return ?
nb_of_samples = 20
x = np.random.uniform(0, 1, nb_of_samples)
t = ?

def gradient(w, x, t, ld): 
  return np.multiply(t, x) + ld * 2 * w if np.multiply(t, np.dot(x, w)) < 1 else ld * 2 * w
def delta_w(w, x, t, learning_rate):
    return learning_rate * gradient(w, x, t)

w = 0.1; learning_rate = 0.1; nb_of_iterations = 10;
w_cost = [(w, cost(nn(x, w), t))]
for i in range(nb_of_iterations):
    w = w - delta_w(w, x, t, learning_rate)
    w_cost.append((w, cost(nn(x, w), t)))
```

<p class="fragment current-only" data-code-focus="1">
  `$ $`
</p>

***

<!-- .slide: style="font-size:24px; text-align: left" -->

#### SVM: Pros and Cons

Pros:

- Effective when 
  - `$ D_\text{features} $` is high-dimensional
  - `$ D_\text{features} \gt N_\text{samples}$`
- Memory Efficient:
  - uses support vectors in the decision function -- a subset of training points
- Versatile:
  - different Kernel functions can be specified for the decision function.

Cons:

- `$ D_\text{features} \gg N_\text{samples}$`
  - Over-fitting
  - Choosing Kernel `$ \phi(x, y) $`and Regularization `$ C $` is crucial 
- No probability estimates
  - Except using an expensive k-fold CV.
- Complexity:
  - `$ \mathcal{O}(D_\text{features} N_\text{samples}^{2 \sim 3}) $`

[scikit-learn User Guide: Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html)  <!-- .element: class="figcaption" -->

---

### Tree Based Learning Algorithms

***

#### Tree

![](https://www.analyticsvidhya.com/wp-content/uploads/2016/04/111.png) [A Complete Tutorial on Tree Based Modeling from Scratch (in R & Python)](https://www.analyticsvidhya.com/blog/2016/04/complete-tutorial-tree-based-modeling-scratch-in-python/) <!-- .element: class="figcaption" -->

***

#### Decision Trees

![](images/decision_tree_for_credit.png) [A Practical Guide to Tree Based Learning Algorithms](https://sadanand-singh.github.io/posts/treebasedmodels/) <!-- .element: class="figcaption" -->

***

#### Decision Trees with Details

![](http://scikit-learn.org/stable/_images/iris.svg)   <!-- .element: style="height: 450px" --> [scikit-learn User Guide: Decision Trees](http://scikit-learn.org/stable/modules/tree.html)   <!-- .element: class="figcaption" -->

***

#### Decision Surface

![](http://efavdb.com/wp-content/uploads/2015/03/tree1.jpg) [Machine Learning Methods: Decision trees and forests](http://efavdb.com/notes-on-trees/) <!-- .element: class="figcaption" -->


***

#### Decision surface of a decision tree on the iris dataset

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_iris_0011.png) <!-- .element: style="height: 450px" --> [scikit-learn User Guide: Plot the decision surface of a decision tree on the iris dataset](http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html)   <!-- .element: class="figcaption" -->

***

#### Decision Tree Regression 

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png)  <!-- .element: style="height: 450px" --> [scikit-learn User Guide: Decision Tree Regression](http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html)   <!-- .element: class="figcaption" -->

***

#### Tree algorithms

<table style="width: 80%; overflow-y: scroll">
  <tr>
    <th>Algorithm</th>
    <th>Features</th> 
  </tr>
  <tr>
    <td>ID3 <small>(Iterative Dichotomiser 3)</small></td>
    <td style="font-size: 24px">
- categorical feature <br />
- yield the largest information gain  <br />
- a pruning step is usually applied for generalization  <br />
    </td>
  </tr>
  <tr class="">
    <td>C4.5</td>
    <td style="font-size: 24px">
- supports numerical variables by partitioning into a discrete set of intervals  <br />
- converts the trained trees into sets of if-then rules  <br />
- pruning by removing precondition
    </td>
  </tr>
  <!-- <tr class="">
    <td>C5.0</td>
    <td style="font-size: 24px">
- proprietary  <br />
- less memory  <br />
- smaller rulesets  <br />
- more accurate  <br />
    </td>
  </tr> -->
  <tr class="">
    <td>CART <small>(Classification and Regression Trees)</small></td>
    <td style="font-size: 24px">
- similar to C4.5  <br />
- supports numerical target variables (regression)  <br />
- does not compute rule sets  <br />
    </td>
  </tr>
</table>

---

### Nearest Neighbors

***

#### K-nearest Neighbors

![](https://github.com/rasbt/python-machine-learning-book/blob/master/code/ch03/images/03_20.png?raw=true)   <!-- .element: class="img-450" --> [Python Machine Learning: Chapter 3 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch03/ch03.ipynb)  <!-- .element: class="figcaption" -->

***

#### Nearest Neighbors Classification

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_classification_001.png)  <!-- .element: class="img-left" -->

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_classification_002.png)  <!-- .element: class="img-right" -->

[scikit-learn User Guide: Nearest Neighbors Classification Example](http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)  <!-- .element: class="figcaption" -->

***

#### Nearest Neighbors Regression

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_regression_001.png)

[scikit-learn User Guide: Nearest Neighbors regression Example](http://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html)  <!-- .element: class="figcaption" -->



***

#### Nearest Centroid Classification

![](https://image.slidesharecdn.com/luccasummerschooltextmininghandouts-121112082043-phpapp02/95/text-and-text-stream-mining-tutorial-23-638.jpg?cb=1352708688)  <!-- .element: style="float:left; width: 49%" -->

![](https://image.slidesharecdn.com/luccasummerschooltextmininghandouts-121112082043-phpapp02/95/text-and-text-stream-mining-tutorial-24-638.jpg?cb=1352708688)  <!-- .element: style="float:right; width: 49%" -->

[Text and text stream mining tutorial](https://www.slideshare.net/mgrcar/text-and-text-stream-mining-tutorial-15137759)  <!-- .element: class="figcaption" -->

***

#### Nearest Neighbor Algorithms

<table style="width: 80%; overflow-y: scroll; font-size: 28px">
  <tr>
    <th>Algorithm</th>
    <th>Complexity</th> 
  </tr>
  <tr>
    <td>Brute Force</td>
    <td>`$ \mathcal{O}(D_\text{features} N_\text{samples}^2) $`</td>
  </tr>
  <tr class="">
    <td>K-D Tree</td>
    <td>`$ \begin{cases} \mathcal{O}( D_\text{features} \log{N_\text{samples}} ) & \mbox{if } D_\text{features} \sim 20 \\[1ex] \mathcal{O}(D_\text{features} N_\text{samples}) & \mbox{if } D_\text{features} \gg 20 \end{cases} $`</td>
  </tr>
  <tr class="">
    <td>Ball Tree</td>
    <td>`$ \mathcal{O}( D_\text{features} \log{N_\text{samples} )} $`</td>
  </tr>
</table>

***

<!-- .slide: data-background-iframe="http://vision.stanford.edu/teaching/cs231n-demos/knn/" data-background-interactive -->

---

### Other Probability Related Methods

***

<!-- .slide: style="font-size:36px; text-align: left" -->

####  Gaussian process (GP)

A GP is the generalization of a Gaussian distribution to a distribution over functions, instead of random variables.

Just as a Gaussian distribution is completely specified by its mean `$ \mu $` and variance `$ \sigma $`, 

a GP is completely specified by its **mean function** `$ m(x) $`, and covariance function `$ k(x,x′) $`.

***

#### Gaussian Process Classification (GPC) on Iris dataset

![](http://scikit-learn.org/stable/_images/sphx_glr_plot_gpc_iris_001.png)

[scikit-learn User Guide: Gaussian process classification (GPC) on iris dataset](http://scikit-learn.org/stable/auto_examples/neighbors/plot_nearest_centroid.html)  <!-- .element: class="figcaption" -->

***

#### Bayes’ theorem

`$$
\mbox{posterior} = \frac{\mbox{prior} \times \mbox{likelihood}}{\mbox{evidence}}
$$`

`$$
P(y \mid x_1, \dots, x_n) = \frac{P(y) \times P(x_1, \dots x_n \mid y)} {P(x_1, \dots, x_n)}
$$`

***

#### Naive Bayes

Using Naive independence assumption:

`$$
P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y)
$$`

`$$
P(y \mid x_1, \dots, x_n) = \frac{P(y) \times \prod_{i=1}^{n} P(x_i \mid y)} {P(x_1, \dots, x_n)}
$$`   <!-- .element: class="fragment current-only" -->

`$$
P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y) 
$$`   <!-- .element: class="fragment current-only" -->

`$$
\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$`   <!-- .element: class="fragment" -->

Note:

https://en.wikipedia.org/wiki/Naive_Bayes_classifier
https://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification

***

[Remove Stepwise Regression and Bayes Regression for now](https://github.com/utensil/slides/commit/9b78364e0c5dff3c2ffee056c60039ac6d0ff6bf#diff-37928793727600d98cc6e6c8a0075033)
https://en.wikipedia.org/wiki/Principal_component_analysis

---

### Links (Temp)

***

<!-- .slide: data-background-iframe="//scikit-learn.org/stable/user_guide.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="http://www.holehouse.org/mlclass/index.html" data-background-interactive -->

***

<!-- .slide: data-background-iframe="http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial" data-background-interactive -->

***

Note:

https://www.zhihu.com/question/26006703

https://github.com/jatinshah/ufldl_tutorial

---

### CV

***

![](https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/ch06/images/06_01.png) [Python Machine Learning: Chapter 6 by Sebastian Raschka](http://nbviewer.jupyter.org/github/rasbt/python-machine-learning-book/blob/master/code/ch06/ch06.ipynb) 

Also 7 Assembling

***

https://en.wikipedia.org/wiki/Correlation_and_dependence
https://en.wikipedia.org/wiki/Instrumental_variables_estimation

***

https://github.com/utensil/julia-playground/blob/master/dl/train_captcha.py
https://keras.io/getting-started/faq/#how-can-i-use-keras-with-datasets-that-dont-fit-in-memory
https://github.com/fchollet/keras/issues/7729
https://github.com/fchollet/keras/issues/1711
https://blog.sigopt.com/posts/common-problems-in-hyperparameter-optimization
https://scikit-optimize.github.io/
https://thuijskens.github.io/2016/12/29/bayesian-optimisation/
https://github.com/thuijskens/bayesian-optimization
https://elitedatascience.com/overfitting-in-machine-learning#how-to-prevent
http://dswalter.github.io/blog/overfitting-regularization-hyperparameters/
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
https://stackoverflow.com/questions/40854232/keras-scikit-learn-using-fit-generator-with-cross-validation
https://github.com/fchollet/keras/issues/4278#issuecomment-264665803

***

#### Learning Rates

![](http://cs231n.github.io/assets/nn3/learningrates.jpeg) [CS231n Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/)  <!-- .element: class="figcaption" -->

***

#### Training/Validation accuracy

![](http://cs231n.github.io/assets/nn3/accuracies.jpeg) [CS231n Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/)  <!-- .element: class="figcaption" -->

---

### Why Go Deep? 

- http://slides.com/beamandrew/deep-learning-101#/
- http://donsoft.io/intro-to-deeplearning

***

<!-- .slide: data-background-iframe="http://cs231n.github.io/assets/conv-demo/index.html" data-background-interactive -->

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

### Math Foundation

- neuroscience narrative, drawing analogies to biology
- representations narrative, centered on transformations of data and the manifold hypothesis
- probabilistic narrative, which interprets neural networks as finding latent variables
- deep learning studies a connection between optimization and functional programming
  - the representations narrative in deep learning corresponds to type theory in functional programming
- http://colah.github.io/posts/2015-09-NN-Types-FP/

***

- http://mathworld.wolfram.com/VectorNorm.html
- https://github.com/utensil/utensil.github.io/issues/15

---

### Libs

- https://simplestatistics.org/docs/

---

### DLib

![](http://dlib.net/ml_guide.svg)

- http://dlib.net/ml.html

---

### Tensorflow

- https://github.com/donnemartin/data-science-ipython-notebooks#tensor-flow-tutorials
- https://github.com/jtoy/awesome-tensorflow
- https://github.com/sjchoi86/Tensorflow-101
- https://github.com/vahidk/EffectiveTensorflow
- https://github.com/nlintz/TensorFlow-Tutorials
- https://github.com/lawlite19/MachineLearning_TensorFlow

---

### Xgboost

- https://xgboost.readthedocs.io/en/latest/model.html
- https://homes.cs.washington.edu/~tqchen/pdf/BoostedTree.pdf
- https://machinelearningmastery.com/gentle-introduction-xgboost-applied-machine-learning/
- http://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf

***

### Higgs Boson

- http://nbviewer.jupyter.org/urls/www.lri.fr/~kegl/HiggsML/StartingKit/startingKitHiggsKegl.ipynb
- https://www.kaggle.com/c/higgs-boson/discussion/13143
- http://opendata.cern.ch/record/329
- http://cs229.stanford.edu/proj2015/017_report.pdf
- http://cs231n.stanford.edu/reports/2016/pdfs/300_Report.pdf
- https://www.researchgate.net/publication/287742463_Searching_for_higgs_boson_decay_modes_with_deep_learning
- https://arxiv.org/pdf/1402.4735.pdf
- https://no2147483647.wordpress.com/2014/09/17/winning-solution-of-kaggle-higgs-competition-what-a-single-model-can-do/

---

### Datasets

- All
  - https://github.com/caesar0301/awesome-public-datasets
  - http://academictorrents.com/
- CV
  - http://deeplearning.net/datasets
- Go
  - https://github.com/yenw/computer-go-dataset
- NLP
  - http://universaldependencies.org/
  - https://github.com/Breakend/DialogDatasets

***

Notebooks:

- http://scikit-learn.org/dev/_static/ml_map.png
- https://github.com/donnemartin/data-science-ipython-notebooks
- https://github.com/jakevdp/PythonDataScienceHandbook

Foundation:

- https://github.com/rushter/MLAlgorithms/
- https://github.com/eriklindernoren/ML-From-Scratch
- https://github.com/ujjwalkarn/Machine-Learning-Tutorials

***

- https://en.wikipedia.org/wiki/Arg_max
- https://tex.stackexchange.com/questions/5223/command-for-argmin-or-argmax
- https://en.wikibooks.org/wiki/LaTeX/Advanced_Mathematics
- https://www.cs.ubc.ca/~schmidtm/Documents/2016_540_Argmax.pdf
- https://raw.githubusercontent.com/scikit-learn/scikit-learn/master/doc/modules/linear_model.rst

---

### Visualization

- http://playground.tensorflow.org
- https://deeplearnjs.org/demos/model-builder/
- http://cs.stanford.edu/people/karpathy/convnetjs/
- http://cs231n.github.io/neural-networks-3/
- http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
- http://colah.github.io/
- https://colah.github.io/posts/2014-07-Understanding-Convolutions/

***

- http://colah.github.io/posts/2015-09-NN-Types-FP/
- http://colah.github.io/posts/2015-01-Visualizing-Representations/
- http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/
- http://colah.github.io/posts/2014-10-Visualizing-MNIST/
- https://zhuanlan.zhihu.com/p/27204291
- https://blog.jakuba.net/2017/05/30/tensorflow-visualization.html

***

- https://github.com/ethereon/netscope
- https://www.zhihu.com/question/26006703/answer/135825424
- http://josephpcohen.com/w/visualizing-cnn-architectures-side-by-side-with-mxnet/

***

http://nbviewer.jupyter.org/github/stephencwelch/Neural-Networks-Demystified/tree/master/
https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
http://www.scipy-lectures.org/intro/numpy/numpy.html#indexing-and-slicing
http://scikit-learn.org/stable/modules/tree.html
https://distill.pub/2017/momentum/
https://github.com/distillpub/post--feature-visualization
https://distill.pub/2017/feature-visualization/
https://bondifrench.github.io/ml-in-js/

---

### Understanding LSTM Networks

- http://colah.github.io/posts/2015-08-Understanding-LSTMs
- https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714

***

<!-- #####.slide: data-background-iframe="//colah.github.io/posts/2015-08-Understanding-LSTMs/" data-background-interactive -->


---

APPENDIX

---

### New Links

***

* https://github.com/karpathy/convnetjs
* http://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html
* http://www.cs.ubc.ca/~van/papers/2016-TOG-deepRL/index.html
* https://github.com/janesjanes/sketchy
* http://scs.ryerson.ca/~aharley/vis/conv/
* http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html
* https://deepart.io/

***

* https://www.youtube.com/playlist?list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU
* https://www.youtube.com/playlist?list=PLE6Wd9FR--EfW8dtjAuPoTuPcqmOV53Fu
* http://swanintelligence.com/first-steps-with-neural-nets-in-keras.html
* http://neuralnetworksanddeeplearning.com/
* http://www.deeplearningbook.org/
* http://boxcar2d.com/index.html

***

* https://www.flickr.com/photos/syntopia/6791724773/in/photostream/
* http://blog.hvidtfeldts.net/index.php/2012/01/knots-and-polyhedra/
* http://blog.mathteachersresource.com/?p=670
* http://www.gitxiv.com/
* http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
* https://github.com/colah/Visualizing-Deep-Learning

***

* https://distill.pub/2017/feature-visualization/
* https://github.com/ChristosChristofidis/awesome-deep-learning#videos-and-lectures
* https://github.com/guillaume-chevalier/Awesome-Deep-Learning-Resources#youtube
* https://github.com/kjw0612/awesome-deep-vision#videos

***

https://hackernoon.com/learning-ai-if-you-suck-at-math-p5-deep-learning-and-convolutional-neural-nets-in-plain-english-cda79679bbe3
http://cs231n.github.io/neural-networks-1/
https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/
http://cs231n.github.io/convolutional-networks/
http://www.deeplearningbook.org/

***

https://nbviewer.jupyter.org/github/makeyourownneuralnetwork/makeyourownneuralnetwork/
https://petewarden.com/2016/02/28/tensorflow-for-poets/
http://blog.otoro.net/2017/01/01/recurrent-neural-network-artist/
https://github.com/hardmaru/rnn-tutorial
https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
https://medium.freecodecamp.org/if-you-want-to-learn-data-science-start-with-one-of-these-programming-classes-fb694ffe780c

***

https://www.docker-cn.com/registry-mirror
https://www.daocloud.io/mirror

***

https://www.mathsisfun.com/algebra/matrix-multiplying.html
https://eli.thegreenplace.net/2015/visualizing-matrix-multiplication-as-a-linear-combination/
https://www.datasciencecentral.com/profiles/blogs/matrix-multiplication-in-neural-networks

***

https://www.quora.com/What-is-the-best-way-to-multiply-two-matrices-in-C++/answer/Jan-Christian-Meyer
http://arainhyy.github.io/#
http://www.cmsoft.com.br/opencl-tutorial/case-study-matrix-multiplication/
https://www.linkedin.com/pulse/lets-untangle-mesh-accelerate-deep-learning-collective-chien-ping-lu

***

http://www.cs.uvm.edu/~icdm/algorithms/10Algorithms-08.pdf
https://twitter.com/demografia_csic/status/536576442495410177
https://startupsventurecapital.com/essential-cheat-sheets-for-machine-learning-and-deep-learning-researchers-efb6a8ebd2e5
https://becominghuman.ai/cheat-sheets-for-ai-neural-networks-machine-learning-deep-learning-big-data-678c51b4b463
http://www.asimovinstitute.org/neural-network-zoo/
https://en.wikipedia.org/wiki/Radial_basis_function
http://thinkzone.wlonk.com/
https://i.pinimg.com/originals/90/a1/87/90a187e216dbf0ba03941a63c17cdcb3.png

***

https://blog.adioma.com/how-to-think-visually-using-visual-analogies-infographic/#_=_
https://medium.com/@shaha/azure-machine-learning-10e0ab2b2c68
https://www.slideshare.net/andrewssobral/machine-learning-for-dummies-without-mathematics
https://www.slideshare.net/aaalm/2016-01-27-a-gentle-and-structured-introduction-to-machine-learning-v105-for-release-58329371
https://docs.microsoft.com/en-us/azure/machine-learning/studio/algorithm-cheat-sheet
https://www.safaribooksonline.com/library/view/deep-learning/9781491924570/ch04.html
https://arxiv.org/pdf/1503.04069.pdf
https://arxiv.org/pdf/1506.00019.pdf
http://proceedings.mlr.press/v37/jozefowicz15.pdf

***

http://nbviewer.jupyter.org/github/aalind0/GAN-auto-write/blob/master/auto_write.ipynb
http://drorbn.net/AcademicPensieve/About.html

***

https://github.com/BAILOOL/DoYouEvenLearn

---

### Links


***

#### ML Top

- [Deep Learning School 2016: Individual Talks](https://www.youtube.com/playlist?list=PLrAXtmErZgOfMuxkACrYnD2fTgbzk2THW)
- [Online Course on Neural Networks](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html)
- https://github.com/fchollet/keras-resources
- https://github.com/lamblin/bayareadlschool
- http://deeplearning.net/datasets

***

- [Deep Learning Summer School, Montreal 2016](https://www.youtube.com/playlist?list=PL5bqIc6XopCbb-FvnHmD1neVlQKwGzQyR)
- [Unsupervised Feature Learning and Deep Learning](http://ufldl.stanford.edu/wiki/index.php/Main_Page)
- [CS224d: Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/)
- [ CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
- https://github.com/baidu-research/ba-dls-deepspeech
- https://github.com/tiny-dnn/tiny-dnn
- https://github.com/dennybritz/reinforcement-learning

***

- https://github.com/Tetrachrome/subpixel
- https://github.com/mxgmn/WaveFunctionCollapse
- https://github.com/SullyChen/Nvidia-Autopilot-TensorFlow
- https://github.com/thoughtfulml/examples-in-python
- http://statweb.stanford.edu/~tibs/ElemStatLearn/
- https://github.com/johnmyleswhite/ML_for_Hackers
- http://lamda.nju.edu.cn/weixs/project/CNNTricks/CNNTricks.html

***

- https://github.com/ZuzooVn/machine-learning-for-software-engineers
- https://github.com/rushter/MLAlgorithms
- http://www.scipy-lectures.org/advanced/image_processing/index.html
- https://zhuanlan.zhihu.com/p/22129946
- http://conflict.lshtm.ac.uk/index.htm
- https://arxiv.org/abs/1404.7828

***

- http://www.jeremydjacksonphd.com/category/deep-learning/
- http://distill.pub/2016/misread-tsne/
- http://playground.tensorflow.org/
- http://projector.tensorflow.org/
- http://ai.berkeley.edu/home.html
- https://en.wikipedia.org/wiki/Topological_data_analysis
- https://www.youtube.com/subscription_manager
- http://rll.berkeley.edu/deeprlcourse/#syllabus ( https://zhuanlan.zhihu.com/p/24721292 )
- https://deepmind.com/blog/wavenet-generative-model-raw-audio/

***

- https://github.com/tensorflow/magenta/blob/master/magenta/reviews/pixelrnn.md
- https://gist.github.com/shagunsodhani/e741ebd5ba0e0fc0f49d7836e30891a7
- https://deepmind.com/blog/differentiable-neural-computers/
- https://deepmind.com/blog/deepmind-round-up-2016/
- https://github.com/phreeza/keras-GAN
- https://github.com/dustinvtran/ml-videos
- https://github.com/oxford-cs-deepnlp-2017/lectures
- https://github.com/rhnvrm/universe-coaster-racer-challenge
- https://github.com/random-forests/tutorials

***

- [Machine Learning Recipes with Josh Gordon](https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal)
- [Reinforcement Learning: An Introduction ](http://webdocs.cs.ualberta.ca/~sutton/book/ebook/the-book.html)
- https://github.com/eriklindernoren/ML-From-Scratch
- https://github.com/stanfordnlp/cs224n-winter17-notes
- https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78#.7tyqpjz6q

***

#### ML

- [【从零单排——数学白痴也能玩机器学习】](https://zhuanlan.zhihu.com/p/19946913)
- [深度学习入门必看的书和论文？有哪些必备的技能需学习？](http://www.zhihu.com/question/31785984)
- [机器视觉、图像处理、机器学习领域相关代码和工程项目和数据集 集合](https://zhuanlan.zhihu.com/p/20787086)
- [CNN(卷积神经网络)、RNN(循环神经网络)、DNN(深度神经网络)的内部网络结构有什么区别？](https://www.zhihu.com/question/34681168)
- [简单解释一下sparse autoencoder, sparse coding和restricted boltzmann machine的关系？](https://www.zhihu.com/question/22906027)
- [在 Deep Learning / Machine Learning 领域，C++ 应如何学习？](https://www.zhihu.com/question/41816315)

***

- [C++ neural network library](https://github.com/clab/cnn)
- [theano](http://www.deeplearning.net/software/theano/)
- [TensorFlow](https://github.com/tensorflow/tensorflow)
- [torch7](https://github.com/torch/torch7)
- [EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES](https://arxiv.org/pdf/1412.6572.pdf) and [
  The Flaw Lurking In Every Deep Neural Net](http://www.i-programmer.info/news/105-artificial-intelligence/7352-the-flaw-lurking-in-every-deep-neural-net.html)

***

- https://github.com/rasbt/python-machine-learning-book
- https://github.com/terryum/awesome-deep-learning-papers
- https://github.com/sjchoi86/dl_tutorials
- http://mp.weixin.qq.com/s?__biz=MzA3MTU0MzcyMQ==&mid=2447602671&idx=1&sn=700ffa7c1a01daa9b5550cc173609925&scene=1&
- https://zhuanlan.zhihu.com/p/22308032
- https://zhuanlan.zhihu.com/p/22107715
- https://www.quora.com/What-are-the-best-resources-to-learn-about-deep-learning

***

#### NLP

- [A Convolutional Neural Network for Modelling Sentences](http://arxiv.org/abs/1404.2188)
- [基于深度学习的自然语言处理在2016年有哪些值得期待的发展？](https://www.zhihu.com/question/40489869)
- [一个玩得停不下来的Google神器：Ngram](https://zhuanlan.zhihu.com/p/20165483)
- [如何评价SyntaxNet？](https://www.zhihu.com/question/46272554)

***

#### CV

- [学习了哪些知识，计算机视觉才算入门？](https://www.zhihu.com/question/26836846)
- [CS231n课程笔记翻译：图像分类笔记](https://zhuanlan.zhihu.com/p/20894041)
- [学习SLAM需要哪些预备知识？](https://www.zhihu.com/question/35186064)

***

Others:
- https://github.com/vic317yeh/One-Click-to-Be-Pro
- https://github.com/geekan/one-python
- https://zhuanlan.zhihu.com/p/22308870
- https://zhuanlan.zhihu.com/p/20092285
- https://zhuanlan.zhihu.com/p/22126107

***

- https://github.com/Kivy-CN/Duke-STA-663-CN
- https://github.com/Kivy-CN/ml-for-humans-zh
- https://github.com/Kivy-CN/Stanford-CS-229-CN
- https://github.com/CuprousIodide/Poinchinski-Memories
- https://foolbox.readthedocs.io/en/latest/user/examples.html
- https://hub.docker.com/r/utensil/dl/

***

- https://github.com/utensil/utensil.github.io/issues/83
- [My Jupyter Notebooks about ML](https://nbviewer.jupyter.org/github/utensil/julia-playground/tree/master/dl/)

---

### Nupic

***

Videos

TODO

***

Papers:

- [Continuous online sequence learning with an unsupervised neural network model](http://arxiv.org/abs/1512.05463)
- [Search for the right paper about HTM](https://discourse.numenta.org/t/search-for-the-right-paper-about-htm/274)

***

Algorithm

- [Spatial Pooler Algorithm Implementation and Pseudocode](http://numenta.com/assets/pdf/biological-and-machine-intelligence/0.4/BaMI-Spatial-Pooler.pdf)
- [Temporal Memory Algorithm Implementation and Pseudocode](http://numenta.com/assets/pdf/biological-and-machine-intelligence/0.4/BaMI-Temporal-Memory.pdf)
- [HTM CLA Flow Chart Diagrams](https://discourse.numenta.org/t/htm-cla-flow-chart-diagrams/713)

***

Implementations

- [NuPIC](https://github.com/numenta/nupic) [<i class="fa fa-code-fork"></i>](https://github.com/numenta/nupic/pulls?utf8=%E2%9C%93&q=author%3Autensil)
- [NuPIC Core](https://github.com/numenta/nupic.core) [<i class="fa fa-code-fork"></i>](https://github.com/numenta/nupic.core/pulls?utf8=%E2%9C%93&q=author%3Autensil)
- [HTM.Julia](https://github.com/htm-community/HTM.Julia)
- [Bare-Bone HTM](http://ifni.co/bbHTM.html)

---

### Further

***

- [Capsule Networks (CapsNets) – Tutorial](https://www.youtube.com/watch?v=pPN8d0E3900&index=14&list=PLKUPUQTsIILCRLQvILCL-VCsg5x3zQiTU)

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

### Draft

***

```python
from time import localtime
 
activities = {8: 'Sleeping', 9: 'Commuting', 17: 'Working',
              18: 'Commuting', 20: 'Eating', 22: 'Resting' }
 
time_now = localtime()
hour = time_now.tm_hour
 
for activity_time in sorted(activities.keys()):
    if hour < activity_time:
        print activities[activity_time]
        break
else:
    print 'Unknown, AFK or sleeping!'
```

***

```python
from time import localtime
 
activities = {8: 'Sleeping', 9: 'Commuting', 17: 'Working',
              18: 'Commuting', 20: 'Eating', 22: 'Resting' }
 
time_now = localtime()
hour = time_now.tm_hour
 
for activity_time in sorted(activities.keys()):
    if hour < activity_time:
        print activities[activity_time]
        break
else:
    print 'Unknown, AFK or sleeping!'
```

<p class="fragment current-only" data-code-focus="1">
  comment 1
</p>
<p class="fragment current-only" data-code-focus="3-4">
  comment 2
</p>
<p class="fragment current-only" data-code-focus="9-14">
  comment 3
</p>

***

<!-- .slide: data-background-video="https://s3.amazonaws.com/static.slid.es/site/homepage/v1/homepage-video-editor.mp4" -->




