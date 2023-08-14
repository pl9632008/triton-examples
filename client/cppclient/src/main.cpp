#include <opencv2/opencv.hpp>
#include "http_client.h"
#include "grpc_client.h"
namespace tc = triton::client;

std::vector< std::string > names = {
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant","stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
	"giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
	"kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
	"knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
	"keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
	"scissors", "teddy bear", "hair drier", "toothbrush"
};

int main(){
	
	bool verbose = false;
	//httpclient
	//std::string url("localhost:8000");
	//std::unique_ptr<tc::InferenceServerHttpClient> client;
	//tc::InferenceServerHttpClient::Create(&client, url, verbose);
	
	//grpcclient
	std:: string url("localhost:8001");
	std::unique_ptr<tc::InferenceServerGrpcClient> client;
	tc::InferenceServerGrpcClient::Create(&client,url,verbose);
	
	cv::Mat img = cv::imread("../zidane.jpg");
	int height = img.rows;
	int width = img.cols;
	
	std::vector<int64_t> img_shape {height,width,3};
	std::vector<int32_t> dims{height,width,3};
	std::vector<int64_t> dims_shape {3};
	std::vector<uint8_t> img_input(height*width*3,0);
	
	for(size_t i = 0 ; i<height*width*3; i++){
	    img_input[i]=img.at<uint8_t>(i) ;
	}
	
	tc::InferInput * input0 ;
	tc::InferInput * input1;
	tc::InferInput::Create(&input0, "ensemble_input", img_shape, "UINT8");
	tc::InferInput::Create(&input1,"ensemble_dims", dims_shape, "INT32");
	
	input0->AppendRaw(
		img_input.data(),
		img_input.size() * sizeof(uint8_t));
	
	input1->AppendRaw(
		reinterpret_cast<uint8_t *>(dims.data()),
		dims.size() * sizeof(int32_t));
	 
	tc::InferRequestedOutput * output0;
	tc::InferRequestedOutput::Create(&output0, "ensemble_output");
	
	std::string model_name = "ensemble_model";
	std::string model_version = "";
	uint32_t client_timeout = 0;
	
	tc::InferOptions options(model_name);
	options.model_version_ = model_version;
	options.client_timeout_ = client_timeout;
	
	tc::Headers http_headers;
	
	std::vector<tc::InferInput*> inputs = {input0, input1};
	std::vector<const tc::InferRequestedOutput*> outputs = {output0};
	
	tc::InferResult* results  ;
	client->Infer(&results, options, inputs, outputs, http_headers);
	
	float * output0_data;
	size_t output0_byte_size;
	
	results->RawData( "ensemble_output",(const uint8_t **) &output0_data , &output0_byte_size);
	
	for(int i = 0 ; i < output0_byte_size/sizeof(output0_data[0]); i+=6){
		
		std::cout<<output0_data[i]<<" "<<output0_data[i+1]<<" "<<output0_data[i+2]<<" "<<
		output0_data[i+3]<<" "<<output0_data[i+4]<<" "<<output0_data[i+5]<<std::endl;	
		
		int x = static_cast<int>( output0_data[i]  );
		int y = static_cast<int>( output0_data[i+1]);
		int w = static_cast<int>( output0_data[i+2]);
		int h = static_cast<int>( output0_data[i+3]);
		float score = output0_data[i+4];
		int label = static_cast<int>( output0_data[i+5]);
		
		cv::rectangle(img,cv::Size(x,y),
				  cv::Size(x+w,y+h),
				  cv::Scalar(100,123,54));
		
		std::string score_fp = std::to_string(score);
		std::string score_out = score_fp.substr(0,score_fp.find(".")+4);
		std::string str = score_out+" "+names[label];
		cv::putText(img, str,cv::Point(x,y),cv::FONT_HERSHEY_SIMPLEX,0.5,cv::Scalar(255,132,0));
	}
	cv::imwrite("/examples/result.jpg",img);

}
