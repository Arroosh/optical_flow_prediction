Jacob Walker (jcwalker@cs.cmu.edu)

This is a modified version of caffe for optical flow prediction. (First release, code
will be given some improvements in the future).

This is an implementation of "Dense Optical Flow from a Static Image," ICCV 2015,
Jacob Walker, Abhinav Gupta, and Martial Hebert

Please look under examples/opticalflow for a demonstration.
Download the trained model on UCF101 from ladoga.graphics.cs.cmu.edu/jcwalker/final.caffemodel to examples/opticalflow

Compile Caffe, and run ./test.sh under examples/opticalflow to generate features for predicted optical flow.

After this, run loadResults.m to visualize the predicted optical flow on the three example images.



# Caffe

Caffe is a deep learning framework developed with cleanliness, readability, and speed in mind.<br />
Consult the [project website](http://caffe.berkeleyvision.org) for all documentation.


Please ask usage questions and how to model different tasks on the [caffe-users mailing list](https://groups.google.com/forum/#!forum/caffe-users).

