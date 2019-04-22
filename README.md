# Scaling Video Analytics on Constrained Edge Nodes
As video camera deployments continue to grow, the need to process large volumes of real-time data strains wide-area network infrastructure. When per-camera bandwidth is limited, it is infeasible for applications such as traffic monitoring and pedestrian tracking to offload high-quality video streams to a datacenter. This paper presents FilterForward, a new edge-to-cloud system that enables datacenter-based applications to process content from thousands of cameras by installing lightweight edge filters that backhaul only relevant video frames. FilterForward introduces fast and expressive per-application “microclassifiers” that share computation to simultaneously detect dozens of events on computationally constrained edge nodes. Only matching events are transmitted to the datacenter. Evaluation on two real-world camera feed datasets shows that FilterForward improves computational efficiency and event detection accuracy for challenging video content while substantially reducing network bandwidth use.

## Paper
*Canel, C., Kim, T., Zhou, G., Li, C., Lim, H., Andersen, D. G., Kaminsky, M., and Dulloor, S. R. Scaling video analytics on constrained edge nodes. In Proceedings of the 2nd SysML Conference (SysML ‘19), Palo Alto, CA, 2019.*

[Published Version](http://www.sysml.cc/doc/2019/197.pdf)  |  Extended Version (coming soon)  |  [Poster](https://github.com/viscloud/filterforward/blob/master/ff_sysml2019_poster.pdf)

## Datasets
Follow [this link](https://drive.google.com/file/d/1ruhlQYBwFrA6Qe2uTL-YmukNvpsL4OaP/view?usp=sharing) to download the *Jackson* and *Roadway* datasets used in the above paper. The *Jackson* dataset is annotated with labels for the *Pedestrian* task and the *Roadway* dataset is annotated with labels for the *People with red* task. Each dataset is composed of a training video and a testing video. See the above paper for details about the datasets.

The video files are at their original frame rate of 30 fps. However, the labels are at 15 fps, and the evaluation in the paper uses 15 fps. For the evaluation, we converted the original video files to 15 fps at various bitrates. We elected to provide the original 30 fps videos so that users can convert them directly to their desired bitrate/frame rate with a single transcode. Providing the videos at 15 fps (which would have required a transcode from 30 fps) would have required users to perform a second transcode, which is not ideal.

The labels files are in the HDF5 format. Each labels file is an array of binary labels, where entry `i` is `1` if frame `i` is an instance of the event, or `0` otherwise. Load a labels file in Python by doing:
```python
import h5py
labels = h5py.File(labels_filepath, 'r')['labels'][:]
```

In some cases, the number of labels is less than the total number of frames in a video. In those cases, discard extra frames from the end of the video.

## Code
***Disclaimer 1: This project is no longer active. We are providing this code "as-is", with no support.***

***Disclaimer 2: We are still in the process of preparing this repository. The code may change.***

FilterForward consists of two codebases (see their `README` files for more info):
1. [`cpp`](https://github.com/viscloud/filterforward/tree/master/cpp): A functional C++ implementation on top of the [SAF video analytics framework](https://github.com/viscloud/saf). We use this for our throughput experiments.
2. [`scripts`](https://github.com/viscloud/filterforward/tree/master/scripts): A set of shell and Python scripts that define, train, and test the microclassifiers and discrete classifiers. We use this for our accuracy and bandwidth experiments. Also included here are scripts for running the throughput experiments using the code in `cpp`.

## Contact
If you have questions, please email [Christopher Canel](https://github.com/ccanel) at ccanel@cmu.edu.
