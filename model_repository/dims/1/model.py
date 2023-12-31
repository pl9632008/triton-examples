
import numpy as np
import json
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        
        # You must parse model_config. JSON string is not parsed here
        model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "dims_output")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])
            
    def execute(self, requests):
        output0_dtype = self.output0_dtype
        responses = []
        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "dims_input")
            
            dims = in_0.as_numpy()
            
            out_tensor_0 = pb_utils.Tensor("dims_output",
                                           dims.astype(output0_dtype))
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0] 
                )
            responses.append(inference_response)
        return responses

    def finalize(self):
  
        print('Cleaning up...')

