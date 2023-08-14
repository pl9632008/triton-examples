import cv2
import numpy as np
import json

import triton_python_backend_utils as pb_utils

class TritonPythonModel:


    def initialize(self, args):


        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "preprocess_output")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
                
    
    def preprocess_img(self,img,input_w,input_h):
        x=0
        y=0
        w=0
        h=0
        height,width,channel = img.shape
        r_w = input_w /(width*1.0)
        r_h = input_h /(height*1.0)
        if r_h > r_w:
            w = input_w
            h = r_w * height
            x = 0
            y = (input_h - h)/2
        else:
            w = r_h * width
            h = input_h
            x = (input_w - w)/2
            y = 0
        re = cv2.resize(img,(int(w),int(h)),interpolation = cv2.INTER_LINEAR)
        out = np.full((input_h,input_w,3),128)
        out[int(y) : int(h) + int(y), int(x) : int(w) + int(x), : ] = re     
        
        out = out.transpose(2,0,1)
        out = out/255.
        
        out = np.expand_dims(out,0)
        
        return out       
     
    
    def execute(self, requests):



        output0_dtype = self.output0_dtype
        

        responses = []


        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "preprocess_input")
            
            image = in_0.as_numpy()
            
            print("preprocess_img = \n",image)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img_out = self.preprocess_img(image,640,640)

            out_tensor_0 = pb_utils.Tensor("preprocess_output",
                                           img_out.astype(output0_dtype))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0] 
                
                )
            responses.append(inference_response)
        return responses

    def finalize(self):
  
        print('Cleaning up...')

