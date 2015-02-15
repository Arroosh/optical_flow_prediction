/*
 * convert_normal.cpp
 *
 *  Created on: Aug 11, 2014
 *      Author: dragon123
 */

#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <set>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::string;

using namespace cv;

bool MyReadImageToDatum(const string& filename,
    const int height, const int width, Datum* datum, bool reSize, bool crop, int scalefloor, int scaleceil)
{
	cv::Mat cv_img, cv_img_origin, tmp;
        cv::Mat cv_flow, cv_flow_origin, tmp2;
        string flowname = filename.substr(0, filename.length()-4);
        flowname = flowname + string(".tif");

	if(reSize)
	{
		cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));	
		cv_flow_origin = cv::imread(flowname, CV_LOAD_IMAGE_COLOR);
		cv::resize(cv_flow_origin, cv_flow, cv::Size(width, height), INTER_NEAREST);

	}

	else
	{
		cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);	
		cv_flow_origin = cv::imread(flowname, CV_LOAD_IMAGE_COLOR);
		cv_img = cv_img_origin;
		cv_flow = cv_flow_origin;
	}

	if(height > cv_img_origin.rows || width > cv_img_origin.cols)
	{
		LOG(ERROR) << "Dimensions larger than image ";
		return false;
	}

	if(crop && !reSize)
	{

		//float theScale = rand()%(scaleceil - scalefloor);

		//theScale = theScale + scalefloor/100.0;

		//cv::resize(cv_img_origin, tmp, cv::Size(cv_img_origin.rows*theScale, cv_img_origin.cols*theScale),
		//		INTER_NEAREST);
		//cv::resize(cv_flow_origin, tmp2, cv::Size(cv_img_origin.rows*theScale, 
		//		cv_img_origin.cols*theScale), INTER_NEAREST);

      		int h_offset = rand() % (cv_img_origin.rows - height);
      		int w_offset = rand() % (cv_img_origin.cols - width);
		cv_img = cv_img_origin(cv::Rect(w_offset, h_offset, width, height));
		cv_flow = cv_flow_origin(cv::Rect(w_offset, h_offset, width, height));
	}


	if (!cv_img.data)
	{
		LOG(ERROR) << "Could not open or find file " << filename;
		return false;
	}

	if (!cv_flow.data)
	{
		LOG(ERROR) << "Could not open or find file " << flowname;
		return false;
	}

	datum->set_channels(3);
	datum->set_height(cv_img.rows);
	datum->set_width(cv_img.cols);
	datum->clear_data();
	datum->clear_float_data();
	datum->clear_multilabel();

	string* datum_string = datum->mutable_data();
	for (int c = 0; c < 3; ++c) {
		for (int h = 0; h < cv_img.rows; ++h) {
		  for (int w = 0; w < cv_img.cols; ++w) {
			datum_string->push_back(
				static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
		  }
		}
	}


	datum_string = datum->mutable_multilabel();
	for (int c = 2; c >= 0; --c) {
		for (int h = 0; h < cv_flow.rows; ++h) {
		  for (int w = 0; w < cv_flow.cols; ++w) {
			datum_string->push_back(
				static_cast<char>(cv_flow.at<cv::Vec3b>(h, w)[c]));
		  }
		}
	}

	return true;
}


int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 5) {
		printf(
				"Convert a set of images to the leveldb format used\n"
						"as input for Caffe.\n"
						"Usage:\n"
						"    convert_imageset ROOTFOLDER/ ANNOTATION DB_NAME"
						" MODE[0-train, 1-val, 2-test] RANDOM_SHUFFLE_DATA[0 or 1, default 1] WIDTH[default 256] HEIGHT[default 256](0 indicates no resize)\n"
						"The ImageNet dataset for the training demo is at\n"
						"    http://www.image-net.org/download-images\n");
		return 0;
	}
	std::ifstream infile(argv[2]);
	std::set<string> fNames;
 	std::vector<string> annos;
	string filename;
	int prop;
	int cc = 0;

	while (infile >> filename)
	{
		if (cc % 1000 == 0)
		LOG(INFO)<<filename;
		cc ++;

		if (fNames.find(filename)== fNames.end())
		{
			fNames.insert(filename);
			filename = filename + std::string("1");
			annos.push_back(filename);
		}
	}

	LOG(INFO)<< "A total of " << annos.size() << " images.";

	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(INFO)<< "Opening leveldb " << argv[3];
	leveldb::Status status = leveldb::DB::Open(options, argv[3], &db);
	CHECK(status.ok()) << "Failed to open leveldb " << argv[3];

	string root_folder(argv[1]);
	Datum datum;
	int count = 0;
	const int maxKeyLength = 256;
	char key_cstr[maxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	int data_size;
	bool data_size_initialized = false;
        int width = 0, height = 0;
	bool resize = 0;
	bool crop = 0;
	int cropNum = 1;
	bool cropAndResize = 0;
	int scalefloor = 100;
	int scaleceil = 100;

        if (argc > 6) width = atoi(argv[6]);
        if (argc > 7) height = atoi(argv[7]);
        if (argc > 8) resize = atoi(argv[8]);
        if (argc > 9) crop = atoi(argv[9]);
        if (argc > 10) cropNum = atoi(argv[10]);
        if (argc > 11) scalefloor = atoi(argv[11]);       
        if (argc > 12) scaleceil = atoi(argv[12]);

        LOG(INFO) << "DIM: " << width << "*" << height;
        
	if(resize)
		LOG(INFO) << " RESIZING " << width << "*" << height;

	if(crop)
		LOG(INFO) << " CROPPING " << width << "*" << height << " " << cropNum << " Times ";
	if(crop && resize)
		cropAndResize = true;


		

	int numImages = annos.size();

	for(int i = 0; i < cropNum - 1; i++)
	{
		for (int anno_id = 0; anno_id < numImages; ++anno_id)
		{
			filename = annos[anno_id];
			filename = filename.substr(0, filename.length()-1);
			filename = filename + std::string("0");
			annos.push_back(filename);
		}
	}

	if (argc < 6 || argv[5][0] != '0') {
		// randomly shuffle data
		LOG(INFO)<< "Shuffling data";
		std::random_shuffle(annos.begin(), annos.end());
	}

	for (int anno_id = 0; anno_id < annos.size(); ++anno_id)
	{

		std::cout << anno_id << " " << annos.size() << std::endl;

		filename = annos[anno_id];

		resize = atoi(filename.substr(filename.length()-1, filename.length()).c_str());

		filename = filename.substr(0, filename.length()-1);

 		if (!MyReadImageToDatum(root_folder + "/" + filename,
				height, width, &datum, resize, crop, scalefloor, scaleceil))
		{
			continue;
		}
		if (!data_size_initialized)
		{
			data_size = datum.channels() * datum.height() * datum.width();
			data_size_initialized = true;
		}
		else
		{
			const string& data = datum.data();
			CHECK_EQ(data.size(), data_size)<< "Incorrect data field size " << data.size();
		}

		// sequential
		snprintf(key_cstr, maxKeyLength, "%07d_%s", anno_id, annos[anno_id].c_str());
		string value;
		// get the value
		datum.SerializeToString(&value);
		batch->Put(string(key_cstr), value);
		if (++count % 1000 == 0)
		{
			db->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR)<< "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
	}
	

	// write the last batch
	if (count % 1000 != 0) {
		db->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR)<< "Processed " << count << " files.";
	}

	delete batch;
	delete db;
	return 0;
}
