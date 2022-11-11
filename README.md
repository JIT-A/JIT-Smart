# JIT-Smart Replication Package



This repository contains the source code and the guidelines for reproducing the paper "JIT-Smart: A Co-Learning Framework for Just-In-Time Defect Prediction and Localization"


The reproduction steps are as follows:

## 🚀Dataset
 Our experiments are based on the large-scale commit-level and line-level dataset [JIT-Defect4J](https://github.com/jacknichao/JIT-Fine), which contains 21 software java projects.




## 🚀 Overview
In this paper, we propose JIT-Smart - a deep learning based and unified JIT-DP and JIT-DL model that can both identify defect-introducing commits (i.e. at commit-level) and locate which code line introduce the defect in the commit (i.e. at line-level). More specifically, the main difference between us compared to previous JIT-DP and JIT-DL research is that we design a novel defect localization network (DLN) specifically for the JIT-DL task, which explicitly introduces the label information of the defective code line for supervised learning. We treat JIT-DP and JIT-DL tasks as a multi-task co-learning process.
![Overview](Overview.png)




## 🚀 Environment Setup



Run the following command under your python environment

```shell
pip install requirements.txt
```



## 🚀 Experiment Result Replication Steps

Note: To minimize the threats to internal validity, we use the source code provided in the corresponding research instead of implementing the compared baselines ourselves. We set the default optimal parameters and operating methods from the corresponding papers to ensure the accuracy of our reproduction. And some evaluation results of the studied baselines are cited from [Ni et al.](https://github.com/jacknichao/JIT-Fine)

### ⭐ RQ1: How effective is JIT-Smart in just-in-time defect prediction task?

Step 1: Run the following two (*.ipynb) files to convert the line-level data annotation format.
  ```shell
./JITSmart/process and label defect code lines/extract defect lines.ipynb

./JITSmart/process and label defect code lines/label defect lines.ipynb
  ```
Step 2: Train and evaluate our JIT-Smart and JIT-Fine in the JIT-DP task.
  ```shell
sh train_jitsmart.sh
sh train_jitfine.sh
  ```




### ⭐ RQ2: How well can JIT-Smart locate defective lines in just-in-time defect localization task?


Step 1: Train and evaluate our JIT-Smart and JIT-Fine in the JIT-DL task.
  ```shell
sh train_jitsmart.sh
sh train_jitfine.sh
  ```


### ⭐ RQ3: What is the accuracy of JIT-Smart compared to the state-of-the-art baseline under the cross-project experimental setting?


Step 1: Generate the cross-project data for JIT-Fine and our JIT-Smart.
  ```shell
python jitsmart cross prj data generate.py
python jitfine cross prj data generate.py
  ```


Step 2: Train and evaluate our JIT-Smart and JIT-Fine in the JIT-DP and JIT-DL tasks under cross-project settings.

  ```shell
sh train_jitsmart_cross_prj.sh
sh train_jitfine_cross_prj.sh
  ```

### ⭐ RQ4: How do the different loss function weight assignments affect the performance of JIT-Smart?



  ```shell
sh train_jitsmart_loss_weight.sh
  ```


