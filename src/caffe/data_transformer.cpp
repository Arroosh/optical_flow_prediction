#include <string>
#include <sstream>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::TransformDataAndLabel(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
				       std::vector< Blob<Dtype>* >& label_list) {
  const string& multilabel = datum.multilabel();
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();
  const int isrand = param_.isrand();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const bool downSample = param_.downsample();
  Dtype scale = param_.scale();

//  cv::Mat cv_img(200,200, CV_8UC3, cv::Scalar::all(0));
 // cv::Mat cv_flow(20,20, CV_8UC3, cv::Scalar::all(0));

 // ostringstream ss;
// ss << batch_item_id;
  
 //           if(batch_item_id % 20 == 18)
//{
//  std::cout << batch_item_id << " ";
//}

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size == 0) {
    LOG(FATAL) << "Current implementation requires crop_size to be "
               << "set if using multi-labels.";
  }

  if(isrand)
  {
	scale = 1.0 + double((rand() % 40) - 20)/100.0;	
  }

  bool doMirror = 0; 
  bool doScale = 0;
    
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
      doMirror = Rand() % 2;
      doScale = Rand() % 2;
    } else {
      doScale = 1;
    }


    if(downSample && doScale) {

      double widthRatio = double(width - 1)/double(crop_size);
      double heightRatio = double(height - 1)/double(crop_size);

      for(int theLabels = 0; theLabels < param_.multilabelscale_size(); theLabels++)
      {  
      		int labelh = 0;
      		int labelw = 0;
      
		Dtype* transformed_labels = label_list[theLabels]->mutable_cpu_data();
  		const int stride = crop_size / param_.multilabelscale(theLabels);
  		const int labelScale = param_.multilabelscale(theLabels);
  		const int numLabels = param_.multilabelscale(theLabels)*param_.multilabelscale(theLabels);

  		if (stride%2 != 0) {
    			LOG(FATAL) << "Current implementation requires crop_size/multilabelscale to be "
               		<< "even.";
  		}

	      for (int c = 0; c < 2; ++c) {
		for (int h = 0; h < crop_size; ++h) {
		  for (int w = 0; w < crop_size; ++w) {
		    int top_index = ((batch_item_id * channels + c) * crop_size + h)
		        * crop_size + w;
		    int data_index = (c * height + round(h*heightRatio)) * width + round(w*widthRatio);
		    Dtype datum_element =
		        static_cast<Dtype>(static_cast<uint8_t>(multilabel[data_index]));
		    transformed_data[top_index] = datum_element;
		  }
		}
	      }
      
	      for (int h = stride/2.0; h < crop_size; h = h + stride) {
		labelw = 0;
		  for (int w = stride/2.0; w < crop_size; w = w + stride) {
		    int top_index = (((batch_item_id) * numLabels)+ (labelh*labelScale) + labelw);
		    int data_index = ((batch_item_id * channels) * crop_size + h)
		        * crop_size + w;
	
		    if(mirror && doMirror)
		    {
		    	top_index = (((batch_item_id) * numLabels)+ (labelh*labelScale) + labelScale - 1 - labelw);
		    	data_index = ((batch_item_id * channels + 1) * crop_size + h)
		        * crop_size + w;
		    }

		    Dtype datum_element =
		        static_cast<Dtype>(static_cast<uint8_t>(transformed_data[data_index]));
		    transformed_labels[top_index] = (datum_element);
	      //       cv_flow.at<cv::Vec3b>(labelh, labelw)[3] = static_cast<uint8_t>(multilabel[data_index]);
	    //       if(batch_item_id % 20 == 18 && !doMirror)
	//{
	//	    std::cout << transformed_labels[top_index] << " ";
	//}
		    labelw++;
		  }
		  labelh++;
		}

    //        if(batch_item_id % 20 == 18 && !doMirror)
//{
 //     std::cout << std::endl;
//}
	}

      // Normal copy

      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;

	    if(mirror && doMirror)
	    {
            top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
	    }

            int data_index = (c * height + round(h*heightRatio)) * width + round(w*widthRatio);



            transformed_data[top_index] = (static_cast<Dtype>(static_cast<uint8_t>(data[data_index])) - mean[data_index]) * 				scale;//DoPixelAverage(height, channels, width, h, c, 
		//	w, heightRatio, widthRatio, data);
//	    if(!doMirror)
//	    {
  //          cv_img.at<cv::Vec3b>(h, w)[c] = (static_cast<Dtype>(static_cast<uint8_t>(data[data_index])));
    //        }

	//    if(doMirror)
       //     {
	//    cv_img.at<cv::Vec3b>(h, 200 - 1 - w)[c] = (static_cast<Dtype>(static_cast<uint8_t>(data[data_index])));
	//    }
          }
        }
      }
 //if(doMirror){
 //    cv::imwrite(ss.str() + std::string(".jpg"), cv_img);}

      return;
    }

    if (!doScale && mirror && doMirror) {
      // Copy mirrored version
      for(int theLabels = 0; theLabels < param_.multilabelscale_size(); theLabels++)
      {  
      		int labelh = 0;
      		int labelw = 0;
      
		Dtype* transformed_labels = label_list[theLabels]->mutable_cpu_data();
  		const int stride = crop_size / param_.multilabelscale(theLabels);
  		const int labelScale = param_.multilabelscale(theLabels);
  		const int numLabels = param_.multilabelscale(theLabels)*param_.multilabelscale(theLabels);

  		if (stride%2 != 0) {
    			LOG(FATAL) << "Current implementation requires crop_size/multilabelscale to be "
               		<< "even.";
  		}

		for (int h = stride/2.0; h < crop_size; h = h + stride) {
		labelw = 0;
		  for (int w = stride/2.0; w < crop_size; w = w + stride) {
		    int top_index = (((batch_item_id) * numLabels)+ (labelh*labelScale) + labelScale - 1 - labelw);
		    int data_index = (1 * height + h + h_off) * width + w + w_off;
		    Dtype datum_element =
		        static_cast<Dtype>(static_cast<uint8_t>(multilabel[data_index]));
		    transformed_labels[top_index] = (datum_element);
		 //    cv_flow.at<cv::Vec3b>(labelh, 20 - 1 - labelw)[3] = static_cast<uint8_t>(multilabel[data_index]);

//	            if(batch_item_id % 20 == 18)
//	{
	//	    std::cout << transformed_labels[top_index] << " ";
//	}
		    labelw++;
		  }
		  labelh++;
		}

 // if(batch_item_id % 20 == 18)
//{
 //     std::cout << std::endl;
//}

	}


      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
           //  cv_img.at<cv::Vec3b>(h, 200 - 1 - w)[c] = static_cast<uint8_t>(data[data_index]);
          }
        }
      }

    //       if(batch_item_id % 20 == 18)
//{
  //    std::cout << std::endl;
//cv::imwrite(ss.str() + std::string(".jpg"), cv_img);
//return;
//}

    } 

    else {
      for(int theLabels = 0; theLabels < param_.multilabelscale_size(); theLabels++)
      {  
      		int labelh = 0;
      		int labelw = 0;
      
		Dtype* transformed_labels = label_list[theLabels]->mutable_cpu_data();
  		const int stride = crop_size / param_.multilabelscale(theLabels);
  		const int labelScale = param_.multilabelscale(theLabels);
  		const int numLabels = param_.multilabelscale(theLabels)*param_.multilabelscale(theLabels);

  		if (stride%2 != 0) {
    			LOG(FATAL) << "Current implementation requires crop_size/multilabelscale to be "
               		<< "even.";
  		}

		for (int h = stride/2.0; h < crop_size; h = h + stride) {
		labelw = 0;
		  for (int w = stride/2.0; w < crop_size; w = w + stride) {
		    int top_index = (((batch_item_id) * numLabels)+ (labelh*labelScale) + labelw);
		    int data_index = (0 * height + h + h_off) * width + w + w_off;
		    Dtype datum_element =
		        static_cast<Dtype>(static_cast<uint8_t>(multilabel[data_index]));
		    transformed_labels[top_index] = (datum_element);
	      //       cv_flow.at<cv::Vec3b>(labelh, labelw)[3] = static_cast<uint8_t>(multilabel[data_index]);
	//            if(batch_item_id % 20 == 18)
	//{
	//	    std::cout << transformed_labels[top_index] << " ";
	//}
		    labelw++;
		  }
		  labelh++;
		}

 //          if(batch_item_id % 20 == 18)
//{
  //    std::cout << std::endl;
//}
	}

      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean[data_index]) * scale;
        //     cv_img.at<cv::Vec3b>(h, w)[c] = static_cast<uint8_t>(data[data_index]);
          }
        }
      }

  //         if(batch_item_id % 20 == 18)
//{
  //    std::cout << std::endl;
//cv::imwrite(ss.str() + std::string(".jpg"), cv_img);
//return;
//}

    }
   //cv::imwrite(ss.str() + std::string(".jpg"), cv_img);
  // cv::imwrite(ss.str() + std::string(".tif"), cv_flow);

}


/*template <typename Dtype>
Dtype DataTransformer<Dtype>::DoPixelAverage(int& height, int& channel, int& width,
int& h, int& c, int& w, double& heightRatio, double& widthRatio, const string& data) {

   if((w*widthRatio) > 0 && w*widthRatio < (width - 1))
   {
   	return static_cast<Dtype>(static_cast<uint8_t>(data[(c * height + h*heightRatio) * width + w*widthRatio]))/3.0 +
   	static_cast<Dtype>(static_cast<uint8_t>(data[(c * height + h*heightRatio) * width + w*widthRatio - 1]))/3.0 +
   	static_cast<Dtype>(static_cast<uint8_t>(data[(c * height + h*heightRatio) * width + w*widthRatio + 1]))/3.0;
   }
   
   else
   {
        return static_cast<Dtype>(static_cast<uint8_t>(data[(c * height + h*heightRatio) * width + w*widthRatio]))
   } 

}*/

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
