paramBall = [];

global paramBall

paramBall.vidDir = '/home/jcwalker/OptFlowBigDataLong/'
paramBall.progDir = '/nfs/hn48/jcwalker/caffe/ImprovedOpticalFlow/improved_trajectory_release/release/'
paramBall.framesToSample = 5
paramBall.movLength = 30
paramBall.buildPanAvi = false
paramBall.optFlowLength = 30
paramBall.randPixSample = 500
paramBall.vectorClusters = 40
paramBall.caffeProgDir = '/nfs/hn48/jcwalker/caffe/'
paramBall.caffeDataDir =  '/home/jcwalker/OptFlowBigDataLongCaffe/'
paramBall.labelDim = 20
paramBall.sampleDim = 200
paramBall.train_net = '/nfs/hn48/jcwalker/caffe/examples/opticalflow/opt_train_coarse_xavier.prototxt'
paramBall.test_net = '/nfs/hn48/jcwalker/caffe/examples/opticalflow/opt_test_coarse_xavier.prototxt'
paramBall.base_lr = 0.00001
paramBall.lr_policy = 'step'
paramBall.gamma = 0.1
paramBall.stepsize = 200000
paramBall.display = 5
paramBall.max_iter = 450000
paramBall.momentum = 0.9
paramBall.weight_decay = 0.0005
paramBall.snapshot = 5000
paramBall.snapshot_prefix = '/home/jcwalker/OptFlowBigDataLongCaffe/OptFlow_'
paramBall.solver_mode = 'GPU'
paramBall.device_id = 2
paramBall.num_filters = 96
paramBall.filter_size = 11
paramBall.secLength = 5
paramBall.getSingleFlow = true
paramBall.canonicalSize = [240 320]
