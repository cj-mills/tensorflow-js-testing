
// The mean of the ImageNet dataset used to train the model
const mean = [0.485, 0.456, 0.406];
// The standard deviation of the ImageNet dataset used to train the model
const std_dev = [0.229, 0.224, 0.225];

//The softmax transforms values to be between 0 and 1
function softmax(resultArray) {
    // Get the largest value in the array.
    const largestNumber = Math.max(...resultArray);
    // Apply exponential function to each result item subtracted by the largest number, use reduce to get the previous result number and the current number to sum all the exponentials results.
    const sumOfExp = resultArray.map((resultItem) => Math.exp(resultItem - largestNumber)).reduce((prevNumber, currentNumber) => prevNumber + currentNumber);
    //Normalizes the resultArray by dividing by the sum of all exponentials; this normalization ensures that the sum of the components of the output vector is 1.
    return resultArray.map((resultValue, index) => {
        return Math.exp(resultValue - largestNumber) / sumOfExp;
    });
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

function generate_grid_strides(height, width, strides = [8, 16, 32]) {

    let grid_strides = new Array();

    // Iterate through each stride value
    for (const stride of strides) {
        // Calculate the grid dimensions
        let grid_height = Math.floor(height / stride);
        let grid_width = Math.floor(width / stride);
        console.log(`Gride: ${grid_height} x ${grid_width}`);

        // Store each combination of grid coordinates
        for (let g1 = 0; g1 < grid_height; g1++) {
            for (let g0 = 0; g0 < grid_width; g0++) {
                grid_strides.push({ 'grid0': g0, 'grid1': g1, 'stride': stride });
            }
        }
    }

    return grid_strides;
}

function generate_yolox_proposals(model_output, proposal_length, grid_strides, bbox_conf_thresh = 0.3) {

    let proposals = new Array();

    // Obtain the number of classes the model was trained to detect
    let num_classes = proposal_length - 5;

    for (let anchor_idx = 0; anchor_idx < grid_strides.length; anchor_idx++) {
        // Get the current grid and stride values
        let grid0 = grid_strides[anchor_idx]['grid0'];
        let grid1 = grid_strides[anchor_idx]['grid1'];
        let stride = grid_strides[anchor_idx]['stride'];

        // Get the starting index for the current proposal
        let start_idx = anchor_idx * proposal_length;

        // Get the coordinates for the center of the predicted bounding box
        let x_center = (model_output[start_idx + 0] + grid0) * stride;
        let y_center = (model_output[start_idx + 1] + grid1) * stride;

        // Get the dimensions for the predicted bounding box
        let w = Math.exp(model_output[start_idx + 2]) * stride;
        let h = Math.exp(model_output[start_idx + 3]) * stride;

        // Calculate the coordinates for the upper left corner of the bounding box
        let x0 = x_center - w * 0.5;
        let y0 = y_center - h * 0.5;

        // Get the confidence score that an object is present
        let box_objectness = model_output[start_idx + 4];

        // Initialize object struct with bounding box information
        let obj = { 'x0': x0, 'y0': y0, 'width': w, 'height': h, 'label': 0, 'prob': 0 };

        // Find the object class with the highest confidence score
        for (let class_idx = 0; class_idx < num_classes; class_idx++) {
            // Get the confidence score for the current object class
            let box_cls_score = model_output[start_idx + 5 + class_idx];
            // Calculate the final confidence score for the object proposal
            let box_prob = box_objectness * box_cls_score;

            // Check for the highest confidence score
            if (box_prob > obj['prob']) {
                obj['label'] = class_idx;
                obj['prob'] = box_prob;
            }

        }

        // Only add object proposals with high enough confidence scores
        if (obj['prob'] > bbox_conf_thresh) { proposals.push(obj); }
    }

    // Sort the proposals based on the confidence score in descending order
    proposals.sort(function (a, b) {
        return parseFloat(b.prob) - parseFloat(a.prob);
    })
    return proposals
}

function calc_union_area(a, b) {
    let x = Math.min(a.x0, b.x0);
    let y = Math.min(a.y0, b.y0);
    let w = Math.max(a.x0 + a.width, b.x0 + b.width) - x;
    let h = Math.max(a.y0 + a.height, b.y0 + b.height) - y;
    return w * h;
}

function calc_inter_area(a, b) {
    let x = Math.max(a.x0, b.x0);
    let y = Math.max(a.y0, b.y0);
    let w = Math.min(a.x0 + a.width, b.x0 + b.width) - x;
    let h = Math.min(a.y0 + a.height, b.y0 + b.height) - y;
    return w * h;
}

function nms_sorted_boxes(proposals, nms_thresh = 0.45) {
    let proposal_indices = new Array();

    // Iterate through the object proposals
    for (let i = 0; i < proposals.length; i++) {
        const a = proposals[i];

        let keep = true;

        // Check if the current object proposal overlaps any selected objects too much
        for (const j of proposal_indices) {
            const b = proposals[j];
            // console.log(b.x0);

            // Calculate the area where the two object bounding boxes overlap
            let inter_area = calc_inter_area(a, b);

            // Calculate the union area of both bounding boxes
            let union_area = calc_union_area(a, b);

            // Ignore object proposals that overlap selected objects too much
            if (inter_area / union_area > nms_thresh) { keep = false; }
        }

        // Keep object proposals that do not overlap selected objects too much
        if (keep) { proposal_indices.push(i); }
    }


    return proposal_indices;
}


let model;

// import { IMAGENET_CLASSES } from './imagenet_classes.js';
import { HAGRID_CLASSES } from './hagrid_classes.js';

// use an async context to call onnxruntime functions.
async function main() {

    tf.setBackend('webgl');
    console.log(`Tensorflow.js backend: ${tf.getBackend()}`);

    var image = document.getElementById('image');
    console.log(`Image Shape: ${image.width} x ${image.height}`);
    var div = document.createElement("DIV");
    div.id = 'output_text';
    div.innerHTML = `Image Source: ${image.src}`;
    document.body.appendChild(div);

    // var model_dir = './models/hagrid-sample-250k-384p-convnext_nano-opset15-tfjs';
    // var model_dir = './models/hagrid-sample-250k-384p-resnet18-opset15-tfjs';
    // var model_dir = './models/hagrid-sample-250k-384p-mobilenetv2_100-opset15-tfjs';
    // var model_dir = './models/hagrid-classification-512p-no_gesture-convnext_nano-opset15-tfjs';
    // var model_dir = './models/hagrid-classification-512p-no_gesture-mobilenetv2_050-opset15-tfjs';
    // var model_dir = './models/hagrid-classification-512p-no-gesture-resnet18-opset15-tfjs';
    // var model_dir = './models/hagrid-classification-512p-no_gesture-convnext_nano-opset15-tfjs-fp32';
    // var model_dir = './models/hagrid-classification-512p-no_gesture-convnext_nano-opset15-tfjs-fp16';
    // var model_dir = './models/hagrid-classification-512p-no_gesture-convnext_nano-opset15-tfjs-uint16';
    // var model_dir = './models/hagrid-classification-512p-no_gesture-convnext_nano-opset15-tfjs-uint8';
    var model_dir = './models/hagrid-sample-250k-384p-YOLOX-tfjs-uint8-ch-last';
    // var model_dir = './models/hagrid-sample-250k-384p-YOLOX-tfjs-fp32-ch-last';
    var model_path = `${model_dir}/model.json`;

    document.getElementById('output_text').innerHTML += `<br>Loading model...`;
    model = await tf.loadGraphModel(model_path, { fromTFHub: false });

    const input_shape = model.inputs[0].shape;
    const height = image.height;
    const width = image.width;
    const grid_strides = generate_grid_strides(height, width);
    // grid_strides.forEach(e => {
    // console.log(e);
    // })
    // console.log(`Gride Strides: ${grid_strides}`);
    // const height = input_shape[1] == -1 ? image.height : input_shape[1];
    // const width = input_shape[2] == -1 ? image.width : input_shape[2];
    console.log(`Input Shape: ${model.inputs[0].shape}`);

    // Warmup the model when using WebGL backend.
    if (tf.getBackend() == 'webgl') {
        document.getElementById('output_text').innerHTML += `<br>Warming up webgl backend...`;
        for (let index = 0; index < 20; index++) {
            tf.tidy(() => {
                // Channels-last format
                // console.log("Here");
                model.predict(tf.zeros([1, height, width, 3])).dispose();
                // model.predict(tf.zeros([1, height + 1, width + 1, 3])).dispose();
            });
        }
    }

    // return;

    var canvas = document.createElement("CANVAS");
    var context = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0);
    var imageData = context.getImageData(0, 0, image.width, image.height);

    // Get buffer data from image.
    var imageBufferData = imageData.data;

    document.getElementById('output_text').innerHTML += `<br>Performing inference...`;
    let inference_start;
    let preprocess_end;
    const preprocess_start = new Date();
    const outputData = tf.tidy(() => {

        // Channels-last format
        const [input_array] = new Array(new Array());
        for (let i = 0; i < imageBufferData.length; i += 4) {
            input_array.push(((imageBufferData[i] / 255.0) - mean[0]) / std_dev[0]);
            input_array.push(((imageBufferData[i + 1] / 255.0) - mean[1]) / std_dev[1]);
            input_array.push(((imageBufferData[i + 2] / 255.0) - mean[2]) / std_dev[2]);
        }

        const float32Data = Float32Array.from(input_array);
        // Reshape to a single-element batch so we can pass it to predict.
        const shape = [1, height, width, 3];

        // Initialize input tensor
        const input_tensor = tf.tensor(float32Data, shape, 'float32');
        preprocess_end = new Date();

        // Make a prediction through model.
        inference_start = new Date();
        return model.predict(input_tensor);
    });
    const inference_time = new Date() - inference_start;
    const preprocess_time = preprocess_end - preprocess_start;
    console.log(outputData.shape);
    // console.log(outputData.array().then(array => console.log(array[0])));
    // const output = await outputData.array();
    const output = await outputData.data();
    console.log(output);
    let proposals = generate_yolox_proposals(output, outputData.shape[2], grid_strides, 0.5);

    const proposal_indices = nms_sorted_boxes(proposals);
    for (let i = 0; i < proposals.length; i++) {
        if (proposal_indices.includes(i)) {
            console.log(proposals[i]);
            context.beginPath();
            context.lineWidth = 3;
            context.strokeStyle = 'green';
            // console.log(width - proposals[i].x0);
            context.strokeRect(proposals[i].x0, proposals[i].y0, proposals[i].width, proposals[i].height);
        }
    }
    // const output = await outputData.data();
    // console.log(output);
    // var results = softmax(Array.prototype.slice.call(output));
    // console.log(results);
    // var index = argMax(Array.prototype.slice.call(output));

    // read from results
    // var pred_string = `${HAGRID_CLASSES[index]} ${(results[index] * 100.0).toPrecision(5)}%`;
    // document.getElementById('output_text').innerHTML += `<br>Predicted class index: ${pred_string}`;
    document.getElementById('output_text').innerHTML += `<br>Preprocess Time: ${preprocess_time}ms`;
    document.getElementById('output_text').innerHTML += `<br>Inference Time: ${inference_time}ms`;
    document.body.appendChild(canvas);
}

main();
