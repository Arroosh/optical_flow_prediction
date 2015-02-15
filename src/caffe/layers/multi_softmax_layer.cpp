// Copyright 2014 BVLC and contributors.
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe
{

template<typename Dtype>
void MultiSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top)
{

	CHECK_EQ(bottom.size(), 1) << "Softmax Layer takes a single blob as input.";
	CHECK_EQ(top->size(), 1) << "Softmax Layer takes a single blob as output.";
	(*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
	sum_multiplier_.Reshape(1, bottom[0]->channels(), bottom[0]->height(),
			bottom[0]->width());
	Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
	for (int i = 0; i < sum_multiplier_.count(); ++i)
	{
		multiplier_data[i] = 1.;
	}
	//wxl
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	int dimClass;

	if(this->layer_param_.mult_softmax_param().num_label())
	{
		dimClass = this->layer_param_.mult_softmax_param().num_label();
		numLabels = dimClass;	
	}	
	else{
		dimClass = numLabels;
	}

	int imgSize = dim / dimClass;
	scale_.Reshape(bottom[0]->num() * imgSize, 1, 1, 1);
}

template<typename Dtype>
void MultiSoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top)
{
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = (*top)[0]->mutable_cpu_data();
	Dtype* scale_data = scale_.mutable_cpu_data();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
	int dimClass = numLabels;

	if(this->layer_param_.mult_softmax_param().num_label())
	{
		dimClass = this->layer_param_.mult_softmax_param().num_label();
		numLabels = dimClass;	
	}	
	else{
		dimClass = numLabels;
	}

	int imgSize = dim / dimClass;
	memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());
	// we need to subtract the max to avoid numerical issues, compute the exp,
	// and then normalize.

	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
			scale_data[i * imgSize + j] = bottom_data[i * dim + j * dimClass];
			for(int k = 0; k < dimClass; ++k)
				scale_data[i * imgSize + j] = max(scale_data[i * imgSize + j], bottom_data[i * dim + j * dimClass + k]);
		}
	}

	// subtraction
	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
			for(int k = 0; k < dimClass; ++k)
				top_data[i * dim + j * dimClass + k] -= scale_data[i * imgSize + j];
		}
	}
	// Perform exponentiation
	caffe_exp<Dtype>(num * dim, top_data, top_data);

	//sum after exp
	for (int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
                        scale_data[i * imgSize + j] = 0;
			for(int k = 0; k < dimClass; ++k)
				scale_data[i * imgSize + j] += top_data[i * dim + j * dimClass + k];
		}
	}
	// Do division

	for(int i = 0; i < num; ++i)
	{
		for (int j = 0; j < imgSize; ++j)
		{
			caffe_scal<Dtype>(dimClass, Dtype(1.) / scale_data[i * imgSize + j], top_data + i * dim + j * dimClass);
		}
	}
}

template<typename Dtype>
void MultiSoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(MultiSoftmaxLayer);
#endif

INSTANTIATE_CLASS(MultiSoftmaxLayer);

}  // namespace caffe
