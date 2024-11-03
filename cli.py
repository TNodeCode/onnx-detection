import os
import cv2
import glob
import onnxruntime as ort
import click
import pandas as pd
from src.helper import read_image, image_to_tensor, get_bboxes, annotate_image


@click.group()
def cli():
    pass


@cli.command()
@click.option('--image-folder', type=click.Path(exists=True, file_okay=False), required=True, help='Path to the folder containing images.')
@click.option('--model-path', type=click.Path(exists=True), required=True, help='Path to the ONNX model file.')
@click.option('--output-dir', type=click.Path(), required=True, help='Directory where annotated images will be saved.')
@click.option('--batch-size', type=int, default=4, help='Batch size for inference.')
def inference(image_folder, model_path, output_dir, batch_size):
    """Run object detection on images in a folder using an ONNX model."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Prepare CSV file to store bounding boxes
    csv_file_path = os.path.join(output_dir, 'detections.csv')
    detections = []

    # Load image paths
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    n_batches = len(image_paths) // batch_size + (len(image_paths) % batch_size > 0)

    # Load the model in ONNX Runtime
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    for batch in range(n_batches):
        img_stack = []
        original_images = []
        for image_path in image_paths[batch * batch_size:(batch + 1) * batch_size]:
            original_images.append(read_image(image_path))
            img_stack.append(image_to_tensor(original_images[-1]))

        # Run inference
        outputs = session.run(None, {input_name: img_stack})

        for i in range(len(outputs[0])):
            boxes = get_bboxes(outputs[0][i])
            annotate_image(image=original_images[i], boxes=boxes)

            # Display the image with bounding boxes
            cv2.imshow("Detections", original_images[i])
            cv2.waitKey(0)  # Wait for a key press to proceed to the next image
            cv2.destroyAllWindows()

            # Save annotated image
            output_file = os.path.join(output_dir, f'annotated_{os.path.basename(image_paths[batch * batch_size + i])}')
            cv2.imwrite(output_file, original_images[i])

            # Collect detections for CSV
            for box in boxes:
                # Assuming box is in the format (xmin, ymin, xmax, ymax)
                detections.append([os.path.basename(image_paths[batch * batch_size + i]), box[0], box[1], box[2], box[3]])

    # Save detections to CSV
    detections_df = pd.DataFrame(detections, columns=['filename', 'xmin', 'ymin', 'xmax', 'ymax'])
    detections_df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    cli()
