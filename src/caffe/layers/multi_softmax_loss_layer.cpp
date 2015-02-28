// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe
{

template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		vector<Blob<Dtype>*>* top)
{
        LossLayer<Dtype>::LayerSetUp(bottom, top);
	CHECK_EQ(bottom.size(), 2) << "Multi SoftmaxLoss Layer takes two blobs as input.";
	//CHECK_EQ(top->size(), 0) << "Multi SoftmaxLoss Layer takes no blob as output.";
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(bottom[0]);
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 2) {
    // softmax output
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top)
{
	// The forward pass computes the softmax prob values.
	softmax_bottom_vec_[0] = bottom[0];
	softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
	const Dtype* prob_data = prob_.cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	int dimClass = bottom[0]->channels();
	int height = bottom[0]->height();
	int width  = bottom[0]->width();
	int imgSize = height * width;
	Dtype loss = 0;
	for (int i = 0; i < num; ++i)
	{
		for(int j = 0; j < imgSize; j ++)
		{
			int nowlabel = static_cast<int>(label[i * imgSize + j]) - 1;
			loss += -log(max( prob_data[i * dim + nowlabel * imgSize + j], Dtype(FLT_MIN)));
		}

	}

  (*top)[0]->mutable_cpu_data()[0] = loss / num ;
  
  if (top->size() == 2) {
    (*top)[1]->ShareData(prob_);
  }
	//return loss / num;
}

template<typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		vector<Blob<Dtype>*>* bottom)
{
	// Compute the diff

	// Compute the diff
	Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
	const Dtype* prob_data = prob_.cpu_data();
	memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
	const Dtype* label = (*bottom)[1]->cpu_data();
	int num = prob_.num();
	int dim = prob_.count() / num;
	int dimClass = (*bottom)[0]->channels();
	int height = (*bottom)[0]->height();
	int width  = (*bottom)[0]->width();
	int imgSize = height * width;

	for (int i = 0; i < num; ++i)
	{
		for(int j = 0; j < imgSize; j ++)
		{
			int nowlabel = static_cast<int>(label[i * imgSize + j]) - 1;
			bottom_diff[ i * dim + nowlabel * imgSize + j  ] -= 1;
		}
	}
	// Scale down gradient
	Dtype multiple = this->layer_param_.mult_softmax_loss_param().learning_multiple();

	caffe_scal(prob_.count(), Dtype(1)*multiple / num, bottom_diff);
}

INSTANTIATE_CLASS(MultiSoftmaxWithLossLayer);

}  // namespace caffe
