*run tf-trt graph*

On Jetson Nano with enough memory (hint: use text mode), try
```
python3 label_image.py --graph=mobilenetv1/mobilenet_v1_fp16.pb \
--image grace_hopper.jpg --labels labels.txt \
--output_layer=MobilenetV1/Predictions/Reshape_1 \
--input_height=224 --input_width=224
```
or

```
python3 label_image.py --graph=mobilenetv1/mobilenet_v2_fp16.pb \
--image grace_hopper.jpg --labels labels.txt \
--output_layer=MobilenetV1/Predictions/Reshape_1 \
--input_height=224 --input_width=224
```
Inferece time should be around 17 ms.

*converting frozen pb*
```
python3 graphdef_to_tensorrt_fp16.py \
--graph /tmp/mobilenet_v1_1.0_224_frozen.pb \
--output_layer MobilenetV1/Predictions/Reshape_1 \
--output_file mobilenetv1/mobilenet_v1_fp16.pb
```
