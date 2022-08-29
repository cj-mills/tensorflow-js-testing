
// const MOBILENET_MODEL_PATH =
//     // tslint:disable-next-line:max-line-length
//     'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1';

// const IMAGE_SIZE = 224;

// The mean of the ImageNet dataset used to train the model
const mean = [0.485, 0.456, 0.406];
// The standard deviation of the ImageNet dataset used to train the model
const std_dev = [0.229, 0.224, 0.225];

async function init_session(model_path, exec_provider) {
    var return_msg;
    try {
        // create a new session and load the specified model.
        session = await ort.InferenceSession.create(model_path,
            { executionProviders: [exec_provider], graphOptimizationLevel: 'all' });
        return_msg = 'Created inference session.';
    } catch (e) {
        return_msg = `failed to create inference session: ${e}.`;
    }
    return return_msg;
}

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

async function PerformInferenceAsync(session, feeds) {

    const outputData = await session.run(feeds);
    return outputData;
}


let mobilenet;

import { IMAGENET_CLASSES } from './imagenet_classes.js';

// use an async context to call onnxruntime functions.
async function main() {

    tf.setBackend('webgl');
    // tf.setBackend('cpu');
    console.log(`Tensorflow.js backend: ${tf.getBackend()}`);

    var image = document.getElementById('image');
    var div = document.createElement("DIV");
    div.id = 'output_text';
    div.innerHTML = `Image Source: ${image.src}`;
    document.body.appendChild(div);


    var model_dir = './models/imagenet_mobilenet_v2_100_224_classification_3_default_1';
    var model_path = `${model_dir}/model.json`;

    console.log('Loading model...');
    // mobilenet = await tf.loadGraphModel(MOBILENET_MODEL_PATH, { fromTFHub: true });
    mobilenet = await tf.loadGraphModel(model_path, { fromTFHub: false });
    const input_shape = mobilenet.inputs[0].shape;
    const height = input_shape[1];
    const width = input_shape[2];
    // const height = image.height;
    // const width = image.width;
    console.log(`Input Shape: ${mobilenet.inputs[0].shape}`);
    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    console.log('Warming up model...');
    for (let index = 0; index < 50; index++) {
        tf.tidy(() => {
            mobilenet.predict(tf.zeros([1, height, width, 3])).dispose();
        });
    }

    var canvas = document.createElement("CANVAS");
    var context = canvas.getContext('2d');
    canvas.width = image.width;
    canvas.height = image.height;
    context.drawImage(image, 0, 0);
    var imageData = context.getImageData(0, 0, image.width, image.height);

    // 1. Get buffer data from image.
    var imageBufferData = imageData.data;
    // console.log(`RGBA Data: ${imageBufferData}`);

    // console.time('Preprocessing');
    // const [redArray, greenArray, blueArray] = new Array(
    //     new Array(),
    //     new Array(),
    //     new Array());

    // 2. Loop through the image buffer and extract the R, G, and B channels
    // for (let i = 0; i < imageBufferData.length; i += 4) {
    //     redArray.push(((imageBufferData[i] / 255.0) - mean[0]) / std_dev[0]);
    //     greenArray.push(((imageBufferData[i + 1] / 255.0) - mean[1]) / std_dev[1]);
    //     blueArray.push(((imageBufferData[i + 2] / 255.0) - mean[2]) / std_dev[2]);
    //     // skip data[i + 3] to filter out the alpha channel
    // }

    // const float32Data = Float32Array.from(redArray.concat(greenArray).concat(blueArray));
    // console.log(`Float Data: ${float32Data}`);
    // console.timeEnd('Preprocessing');

    // const input = input_tensor.reshape(shape);
    // console.log(input);

    // 5. create the tensor object from onnxruntime-web.
    // const input_tensor = new ort.Tensor("float32", float32Data, [1, 3, image.height, image.width]);
    // const feeds = {};
    // feeds[session.inputNames[0]] = input_tensor;


    console.log('Performing inference...');
    const offset = tf.scalar(127.5);
    let inference_start;
    let preprocess_end;
    const preprocess_start = new Date();
    const outputData = tf.tidy(() => {

        const offset_val = 127.5;
        const [input_array] = new Array(new Array());
        for (let i = 0; i < imageBufferData.length; i += 4) {
            input_array.push(((imageBufferData[i]) - offset_val) / offset_val);
            input_array.push(((imageBufferData[i + 1]) - offset_val) / offset_val);
            input_array.push(((imageBufferData[i + 2]) - offset_val) / offset_val);
            // skip data[i + 3] to filter out the alpha channel
        }

        const float32Data = Float32Array.from(input_array);

        const shape = [1, height, width, 3];
        const input_tensor = tf.tensor(float32Data, shape, 'float32');

        // Reshape to a single-element batch so we can pass it to predict.
        preprocess_end = new Date();

        inference_start = new Date();
        // Make a prediction through mobilenet.
        return mobilenet.predict(input_tensor);
    });
    const preprocess_time = preprocess_end - preprocess_start;
    const inference_time = new Date() - inference_start;
    const output = await outputData.data();
    // console.log(output);
    // results = softmax(Array.prototype.slice.call(output));
    // console.log(results);
    var index = argMax(Array.prototype.slice.call(output));

    // feed inputs and run
    // const start = new Date();
    // var index = await PerformInferenceAsync(session, feeds).then(outputData => {
    //     const output = outputData[session.outputNames[0]];
    //     results = softmax(Array.prototype.slice.call(output.data));
    //     console.log(results);
    //     return argMax(this.results);
    // })
    // const end = new Date();

    // read from results
    document.getElementById('output_text').innerHTML += `<br>Predicted class index: ${IMAGENET_CLASSES[index - 1]}`;
    document.getElementById('output_text').innerHTML += `<br>Preprocess Time: ${preprocess_time}ms`;
    document.getElementById('output_text').innerHTML += `<br>Inference Time: ${inference_time}ms`;
}

main();
