import cv2
import onnxruntime as ort
from src.helper import read_image, image_to_tensor, get_bboxes, annotate_image

# Load the ONNX model
model_path = 'work_dirs/faster_rcnn_spine.onnx'

batch_size = 4
image_paths = ['000001.png']
n_batches = len(image_paths) // batch_size + 1

for batch in range(n_batches):
    img_stack = []
    orignal_images = []
    for image_path in image_paths[batch:batch+batch_size]:
        orignal_images.append(read_image(image_path))
        img_stack.append(image_to_tensor(orignal_images[-1]))

    # Load the model in ONNX Runtime
    session = ort.InferenceSession(model_path)

    # Get input name (assuming the first input)
    input_name = session.get_inputs()[0].name

    # Run inference
    outputs = session.run(None, {input_name: img_stack})

    # Process the output (assuming output is the first one)
    output_name = session.get_outputs()[0].name

    for i in range(outputs[0].shape[0]):
        boxes = get_bboxes(outputs[0][i])
        annotate_image(image=orignal_images[i], boxes=boxes)

        # Display the image with bounding boxes
        cv2.imshow("Detections", orignal_images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite('output.png', orignal_images[i])
