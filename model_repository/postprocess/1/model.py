import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):

        model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "postprocess_output")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])


    def decode(self,prob,height,width):
        
        res = []
 
        for i in prob :
            if i[4] <= 0.5:
                continue
            
            l,r,t,b=0,0,0,0
            r_w = 640/(width*1.0)
            r_h = 640/(height*1.0)

            x = i[0]
            y = i[1]
            w = i[2]
            h = i[3]
            score = i[4]

            if r_h > r_w :
                l = x-w/2.0
                r = x+w/2.0
                t = y-h/2.0 - (640 - r_w * height)/2
                b = y+h/2.0 - (640 - r_w * height)/2
                l/=r_w
                r/=r_w
                t/=r_w
                b/=r_w
            else:
                l = x-w/2.0 - (640 - r_h * width)/2
                r = x+w/2.0 - (640 - r_h * width)/2
                t = y-h/2.0 
                b = y+h/2.0
                l/=r_h
                r/=r_h
                t/=r_h
                b/=r_h

            label_index = np.argmax(i[5:])
 
            res.append([l,t,r-l,b-t,score,label_index])
            
        # res.sort(key=lambda res:res[4],reverse=True)
        res=np.array(res)    

        idx = cv2.dnn.NMSBoxes(res[:,:4], res[:,4], 0.5, 0.5)
        
        boxes = res[idx]

        return boxes
            

    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []
       
        for request in requests:

            in_1 = pb_utils.get_input_tensor_by_name(request, "postprocess_input")

            in_2 = pb_utils.get_input_tensor_by_name(request, "postprocess_dims")            
            
            in_1 = in_1.as_numpy()
            
            in_2 = in_2.as_numpy()
            
            height,width,_ = in_2
            print("in_1 = \n",in_1)
            
             
            print("in_2 = \n",in_2)
            

            cropped_arr = self.decode(in_1[0],height,width)
            
            
            
            print("cropped_arr = \n",cropped_arr)

            # print("height = ",height," width = ",width)
            # print("cropped_arr = ",cropped_arr)
            
            out_tensor_0 = pb_utils.Tensor("postprocess_output",
                                           cropped_arr.astype(output0_dtype))

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0])
            responses.append(inference_response)

        return responses

    def finalize(self):

        print('Cleaning up...')
