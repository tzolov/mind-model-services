/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.mindmodel.services.image.recognition;

import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import io.mindmodel.services.common.GraphicsUtils;
import io.mindmodel.services.common.JsonMapperFunction;
import org.apache.commons.io.IOUtils;

/**
 * @author Christian Tzolov
 */
public class ImageRecognitionExample {

	public static void main(String[] args) throws IOException {

		// MmobileNetV2 models
		String mobilenet_v2_modelUri = "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz#mobilenet_v2_1.4_224_frozen.pb";
		ImageRecognitionService recognitionService = ImageRecognitionService.mobilenetModeV2(
				mobilenet_v2_modelUri,
				224,
				5,
				true);

		// MmobileNetV1 models
		// String mobilenet_v1_modelUri = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz#mobilenet_v1_1.0_224_frozen.pb";
		//ImageRecognitionService recognitionService = ImageRecognitionService.mobilenetModeV1(mobilenet_v1_modelUri, 224, 5, true);

		// Inception models
		//String inception_modelUri = "https://dl.bintray.com/big-data/generic/tensorflow_inception_graph.pb";
		//ImageRecognitionService recognitionService = ImageRecognitionService.inception(inception_modelUri, 224, 5, true);


		// You can use file:, http: or classpath: to provide the path to the input image.
		byte[] inputImage = GraphicsUtils.loadAsByteArray("classpath:/images/giant_panda_in_beijing_zoo_1.jpg");

		List<RecognitionResponse> recognizedObjects = recognitionService.recognize(inputImage);

		// Draw the predicted labels on top of the input image.
		byte[] augmentedImage = new ImageRecognitionAugmenter().apply(inputImage, recognizedObjects);
		IOUtils.write(augmentedImage, new FileOutputStream("./image-recognition/target/image-augmented.jpg"));


		String jsonRecognizedObjects = new JsonMapperFunction().apply(recognizedObjects);
		System.out.println(jsonRecognizedObjects);
	}

}
