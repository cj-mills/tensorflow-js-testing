
// The mean of the ImageNet dataset used to train the model
const mean = [0.485, 0.456, 0.406];
// The standard deviation of the ImageNet dataset used to train the model
const std_dev = [0.229, 0.224, 0.225];


function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}

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

let model;

// import { IMAGENET_CLASSES } from './imagenet_classes.js';
import { HAGRID_CLASSES } from './hagrid_classes.js';

// use an async context to call onnxruntime functions.
async function main() {

    tf.setBackend('webgl');
    console.log(`Tensorflow.js backend: ${tf.getBackend()}`);

    var image = document.getElementById('image');
    var div = document.createElement("DIV");
    div.id = 'output_text';
    div.innerHTML = `Image Source: ${image.src}`;
    document.body.appendChild(div);


    var model_dir = './models/hagrid-sample-250k-384p-convnext_nano-opset15-tfjs';
    // var model_dir = './models/hagrid-sample-250k-384p-resnet18-opset15-tfjs';
    // var model_dir = './models/hagrid-sample-250k-384p-mobilenetv2_100-opset15-tfjs';
    var model_path = `${model_dir}/model.json`;

    document.getElementById('output_text').innerHTML += `<br>Loading model...`;
    model = await tf.loadGraphModel(model_path, { fromTFHub: false });

    const input_shape = model.inputs[0].shape;
    const height = input_shape[1] == -1 ? image.height : input_shape[1];
    const width = input_shape[2] == -1 ? image.width : input_shape[2];
    console.log(`Input Shape: ${model.inputs[0].shape}`);

    // Warmup the model when using WebGL backend.
    if (tf.getBackend() == 'webgl') {
        document.getElementById('output_text').innerHTML += `<br>Warming up webgl backend...`;
        for (let index = 0; index < 50; index++) {
            tf.tidy(() => {
                // Channels-last format
                model.predict(tf.zeros([1, height, width, 3])).dispose();
            });
        }
    }


    var canvas = document.createElement("CANVAS");
    var context = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0);
    var imageData = context.getImageData(0, 0, image.width, image.height);

    // Get buffer data from image.
    var imageBufferData = imageData.data;

    console.log('Performing inference...');
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
        const shape = [1, height, width, 3];


        const input_tensor = tf.tensor(float32Data, shape, 'float32');

        // Reshape to a single-element batch so we can pass it to predict.
        preprocess_end = new Date();

        inference_start = new Date();
        // Make a prediction through model.
        return model.predict(input_tensor);
    });
    const preprocess_time = preprocess_end - preprocess_start;
    const inference_time = new Date() - inference_start;
    const output = await outputData.data();
    // console.log(output);
    var results = softmax(Array.prototype.slice.call(output));
    console.log(results);
    var index = argMax(Array.prototype.slice.call(output));

    // read from results
    document.getElementById('output_text').innerHTML += `<br>Predicted class index: ${HAGRID_CLASSES[index]}`;
    document.getElementById('output_text').innerHTML += `<br>Preprocess Time: ${preprocess_time}ms`;
    document.getElementById('output_text').innerHTML += `<br>Inference Time: ${inference_time}ms`;
}

main();
