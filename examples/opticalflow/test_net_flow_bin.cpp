

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <utility>
#include <algorithm>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <leveldb/db.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"

#include <string>
#include <fstream>

using namespace std;
using namespace cv;
using namespace caffe;
using std::vector;

int CreateDir(const char *sPathName, int beg) {
	char DirName[256];
	strcpy(DirName, sPathName);
	int i, len = strlen(DirName);
	if (DirName[len - 1] != '/')
		strcat(DirName, "/");

	len = strlen(DirName);

	for (i = beg; i < len; i++) {
		if (DirName[i] == '/') {
			DirName[i] = 0;
			if (access(DirName, 0) != 0) {
				CHECK(mkdir(DirName, 0755) == 0)<< "Failed to create folder "<< sPathName;
			}
			DirName[i] = '/';
		}
	}

	return 0;
}

char buf[101000];
int main(int argc, char** argv)
{
	cudaSetDevice(atoi(argv[7]));
	Caffe::set_phase(Caffe::TEST);

	if (argc == 8 && strcmp(argv[7], "CPU") == 0) {
		LOG(ERROR) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	} else {
		LOG(ERROR) << "Using GPU";
		Caffe::set_mode(Caffe::GPU);
	}

	NetParameter test_net_param;
	ReadProtoFromTextFile(argv[1], &test_net_param);
	Net<float> caffe_test_net(test_net_param);
	NetParameter trained_net_param;
	ReadProtoFromBinaryFile(argv[2], &trained_net_param);
	caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

	vector<shared_ptr<Layer<float> > > layers = caffe_test_net.layers();
	//const DataLayer<float> *datalayer = dynamic_cast<const DataLayer<float>* >(layers[0].get());
	//CHECK(datalayer);

	string labelFile(argv[3]);
	int data_counts = 0;
	FILE * file = fopen(labelFile.c_str(), "r");
	while(fgets(buf,100000,file) > 0)
	{
		data_counts++;
	}
	fclose(file);

	vector<Blob<float>*> dummy_blob_input_vec;
	string rootfolder(argv[4]);
	rootfolder.append("/");
	CreateDir(rootfolder.c_str(), rootfolder.size() - 1);
	string folder;
	string fName;

	float output;
	int counts = 0;


	int numLabels = atoi(argv[5]);
	int outputDim = atoi(argv[6]);

	file = fopen(labelFile.c_str(), "r");

	Blob<float>* c1 = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
    int c2 = c1->num();
	int batchCount = std::ceil(data_counts / (floor)(c2));

	string resulttxt = rootfolder + "FlowResult.txt";

	for (int batch_id = 0; batch_id < batchCount; ++batch_id)
	{
		LOG(INFO)<< "processing batch :" << batch_id+1 << "/" << batchCount <<"...";

		const vector<Blob<float>*>& result = caffe_test_net.Forward(dummy_blob_input_vec);
		Blob<float>* bboxs = (*(caffe_test_net.bottom_vecs().rbegin()))[0];
		int bsize = bboxs->num();

		int channel = bboxs->channels();
		int height  = bboxs->height();
		int width   = bboxs->width();
		printf("channel:%d, height:%d, width:%d\n", channel, height, width);

		const Blob<float>* labels = (*(caffe_test_net.bottom_vecs().rbegin()))[1];
		Blob<float> aBlob;
		aBlob.Reshape(1,channel, height, width);
		float* aBlobData = aBlob.mutable_cpu_data();
		float* bBData = bboxs->mutable_cpu_data();

		for (int i = 0; i < bsize && counts < data_counts; i++, counts++)
		{
			char fname[1010];

			std::stringstream theNum;

			theNum << counts;
			fscanf(file, "%s", fname);
			std::string resulttxt = rootfolder + theNum.str() + std::string(".txt");
			FILE * resultfile = fopen(resulttxt.c_str(), "w");
			fprintf(resultfile, "%s ", fname);
			std::string thename(theNum.str());
			thename = rootfolder + thename + std::string(".h5");

 			hid_t file_id_ = H5Fcreate(thename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                        H5P_DEFAULT);

			caffe_copy(channel*height*width,bBData + bboxs->offset(i,0,0,0),aBlobData);

			hdf5_save_nd_dataset(file_id_, "Outputs", aBlob);					
			fclose(resultfile);
			H5Fclose(file_id_);
			fscanf(file, "%s", fname);
		}
	}


	fclose(file);


	return 0;
}
